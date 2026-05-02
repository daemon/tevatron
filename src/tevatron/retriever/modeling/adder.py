import json
import os
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import PreTrainedModel

from .encoder import EncoderModel


ADDER_CONFIG_NAME = "adder_config.json"
ADDER_WEIGHTS_NAME = "adder_model.bin"


class AdderModel(EncoderModel):
    """
    Fixed multi-vector retriever that scores pointwise query/document vectors with LogSumExp.
    """

    def __init__(
            self,
            encoder: PreTrainedModel,
            pooling: str = 'adder',
            normalize: bool = False,
            temperature: float = 1.0,
            num_vectors: int = 32,
            num_layers: int = 2,
            num_heads: Optional[int] = None,
            projection_dim: Optional[int] = None,
            dropout: float = 0.1,
            logsumexp_temperature: float = 1.0,
    ):
        super().__init__(
            encoder=encoder,
            pooling=pooling,
            normalize=normalize,
            temperature=temperature,
        )
        hidden_size = self.config.hidden_size
        num_heads = num_heads or getattr(self.config, "num_attention_heads", 1)
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Adder hidden size ({hidden_size}) must be divisible by num_heads ({num_heads})."
            )

        self.num_vectors = num_vectors
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.projection_dim = projection_dim
        self.dropout = dropout
        self.logsumexp_temperature = logsumexp_temperature

        self.query_tokens = nn.Parameter(torch.empty(num_vectors, hidden_size))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=getattr(self.config, "intermediate_size", hidden_size * 4),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.querying_transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, projection_dim) if projection_dim else nn.Identity()
        self._reset_adder_parameters()

    def _reset_adder_parameters(self):
        nn.init.normal_(self.query_tokens, mean=0.0, std=getattr(self.config, "initializer_range", 0.02))

    def encode_query(self, qry):
        return self._encode_fixed_vectors(qry)

    def encode_passage(self, psg):
        return self._encode_fixed_vectors(psg)

    def _encode_fixed_vectors(self, inputs: Dict[str, Tensor]):
        outputs = self.encoder(**inputs, return_dict=True)
        hidden_states = outputs.last_hidden_state
        batch_size = hidden_states.size(0)
        query_tokens = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        attention_mask = inputs.get("attention_mask")
        memory_key_padding_mask = None
        if attention_mask is not None:
            memory_key_padding_mask = ~attention_mask.bool()

        reps = self.querying_transformer(
            tgt=query_tokens,
            memory=hidden_states,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        reps = self.output_norm(reps)
        reps = self.output_projection(reps)
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    def compute_similarity(self, q_reps, p_reps):
        if q_reps.dim() != 3 or p_reps.dim() != 3:
            raise ValueError("AdderModel expects query and passage reps shaped [batch, num_vectors, hidden].")
        if q_reps.size(1) != p_reps.size(1):
            raise ValueError("AdderModel requires the same number of query and passage vectors.")

        pointwise_scores = torch.einsum("qvd,pvd->qpv", q_reps, p_reps)
        tau = self.logsumexp_temperature
        if tau <= 0:
            raise ValueError("logsumexp_temperature must be positive.")
        return tau * torch.logsumexp(pointwise_scores / tau, dim=-1)

    def get_adder_config(self):
        return {
            "pooling": self.pooling,
            "normalize": self.normalize,
            "temperature": self.temperature,
            "num_vectors": self.num_vectors,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "projection_dim": self.projection_dim,
            "dropout": self.dropout,
            "logsumexp_temperature": self.logsumexp_temperature,
        }

    @classmethod
    def build(cls, model_args, train_args, **hf_kwargs):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if model_args.lora or model_args.lora_name_or_path:
            if train_args.gradient_checkpointing:
                base_model.enable_input_require_grads()
            if model_args.lora_name_or_path:
                lora_config = LoraConfig.from_pretrained(model_args.lora_name_or_path, **hf_kwargs)
                base_model = PeftModel.from_pretrained(
                    base_model,
                    model_args.lora_name_or_path,
                    config=lora_config,
                    is_trainable=True,
                )
            else:
                lora_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=model_args.lora_target_modules.split(','),
                    inference_mode=False,
                )
                base_model = get_peft_model(base_model, lora_config)
        return cls(
            encoder=base_model,
            pooling=model_args.pooling,
            normalize=model_args.normalize,
            temperature=model_args.temperature,
            num_vectors=model_args.adder_num_vectors,
            num_layers=model_args.adder_num_layers,
            num_heads=model_args.adder_num_heads,
            projection_dim=model_args.adder_projection_dim,
            dropout=model_args.adder_dropout,
            logsumexp_temperature=model_args.adder_logsumexp_temperature,
        )

    @classmethod
    def load(
            cls,
            model_name_or_path: str,
            pooling: str = 'adder',
            normalize: bool = False,
            temperature: float = 1.0,
            lora_name_or_path: str = None,
            num_vectors: int = 32,
            num_layers: int = 2,
            num_heads: Optional[int] = None,
            projection_dim: Optional[int] = None,
            dropout: float = 0.1,
            logsumexp_temperature: float = 1.0,
            **hf_kwargs
    ):
        config_path = os.path.join(model_name_or_path, ADDER_CONFIG_NAME)
        if os.path.exists(config_path):
            with open(config_path) as f:
                adder_config = json.load(f)
            pooling = adder_config.get("pooling", pooling)
            normalize = adder_config.get("normalize", normalize)
            temperature = adder_config.get("temperature", temperature)
            num_vectors = adder_config.get("num_vectors", num_vectors)
            num_layers = adder_config.get("num_layers", num_layers)
            num_heads = adder_config.get("num_heads", num_heads)
            projection_dim = adder_config.get("projection_dim", projection_dim)
            dropout = adder_config.get("dropout", dropout)
            logsumexp_temperature = adder_config.get("logsumexp_temperature", logsumexp_temperature)
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
            base_model = PeftModel.from_pretrained(base_model, lora_name_or_path, config=lora_config)
            base_model = base_model.merge_and_unload()

        model = cls(
            encoder=base_model,
            pooling=pooling,
            normalize=normalize,
            temperature=temperature,
            num_vectors=num_vectors,
            num_layers=num_layers,
            num_heads=num_heads,
            projection_dim=projection_dim,
            dropout=dropout,
            logsumexp_temperature=logsumexp_temperature,
        )

        weights_path = os.path.join(model_name_or_path, ADDER_WEIGHTS_NAME)
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
        return model

    def save(self, output_dir: str, state_dict=None, safe_serialization: bool = True):
        os.makedirs(output_dir, exist_ok=True)
        if state_dict is None:
            state_dict = self.state_dict()

        encoder_prefix = "encoder."
        encoder_state_dict = {
            key[len(encoder_prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(encoder_prefix)
        }
        adder_state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith(encoder_prefix)
        }
        self.encoder.save_pretrained(
            output_dir,
            state_dict=encoder_state_dict,
            safe_serialization=safe_serialization,
        )
        with open(os.path.join(output_dir, ADDER_CONFIG_NAME), "w") as f:
            json.dump(self.get_adder_config(), f, indent=2, sort_keys=True)
        torch.save(adder_state_dict, os.path.join(output_dir, ADDER_WEIGHTS_NAME))
