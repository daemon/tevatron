#!/usr/bin/env zsh
set -e
EMBEDDING_OUTPUT_DIR=embeddings-adder-more

run_shard() {
  local s=$1
  local gpu=$2

  CUDA_VISIBLE_DEVICES=$gpu python -m tevatron.retriever.driver.encode \
    --output_dir temp \
    --model_type adder \
    --model_name_or_path retriever-bert-adder-more \
    --query_prefix "Query: " \
    --passage_prefix "Passage: " \
    --bf16 \
    --normalize \
    --per_device_eval_batch_size 256 \
    --query_max_len 32 \
    --passage_max_len 180 \
    --dataset_name Tevatron/msmarco-passage-corpus \
    --dataset_number_of_shards 16 \
    --dataset_shard_index "$s" \
    --encode_output_path "$EMBEDDING_OUTPUT_DIR/corpus.$s.pkl" \
    --attn_implementation sdpa
}

{
  for s in 0 4 8 12; do
    run_shard "$s" 0
  done
} &

{
  for s in 2 6 10 14; do
    run_shard "$s" 1
  done
} &

{
  for s in 1 5 9 13; do
    run_shard "$s" 2
  done
} &

{
  for s in 3 7 11 15; do
    run_shard "$s" 3
  done
} &

wait
