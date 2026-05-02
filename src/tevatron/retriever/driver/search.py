import pickle

import numpy as np
import glob
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
from tqdm import tqdm
import faiss
import torch

from tevatron.retriever.searcher import FaissFlatSearcher

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def search_queries(retriever, q_reps, p_lookup, args):
    if args.batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, args.depth, args.batch_size, args.quiet)
    else:
        all_scores, all_indices = retriever.search(q_reps, args.depth)

    psg_indices = [[str(p_lookup[x]) for x in q_dd] for q_dd in all_indices]
    psg_indices = np.array(psg_indices)
    return all_scores, psg_indices


def get_torch_dtype(dtype):
    return {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }[dtype]


def merge_topk(existing_scores, existing_docids, new_scores, new_docids, depth):
    if existing_scores is None:
        return new_scores, new_docids

    merged_scores = np.concatenate([existing_scores, new_scores], axis=1)
    merged_docids = np.concatenate([existing_docids, new_docids], axis=1)
    k = min(depth, merged_scores.shape[1])
    rows = np.arange(merged_scores.shape[0])[:, None]
    indices = np.argpartition(-merged_scores, kth=k - 1, axis=1)[:, :k]
    scores = merged_scores[rows, indices]
    docids = merged_docids[rows, indices]
    order = np.argsort(-scores, axis=1)
    return np.take_along_axis(scores, order, axis=1), np.take_along_axis(docids, order, axis=1)


def score_adder_shards_on_device(q_reps_np, index_files, args, device_name, show_progress=False, progress_bar=None):
    device = torch.device(device_name)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    score_dtype = get_torch_dtype(args.dtype)
    q_reps = torch.from_numpy(q_reps_np).to(device=device, dtype=score_dtype)
    worker_scores = None
    worker_docids = None

    shard_iterator = index_files
    if show_progress:
        shard_iterator = tqdm(index_files, desc=f'Scoring shards on {device}', disable=args.quiet)

    with torch.inference_mode():
        for index_file in shard_iterator:
            p_reps, p_lookup = pickle_load(index_file)
            p_lookup = np.asarray(p_lookup, dtype=object)
            shard_scores = []
            shard_docids = []

            query_iterator = range(0, q_reps.size(0), args.batch_size)
            if show_progress:
                query_iterator = tqdm(query_iterator, desc='Scoring queries', leave=False, disable=args.quiet)

            for q_start in query_iterator:
                q_batch = q_reps[q_start:q_start + args.batch_size]
                batch_scores = None
                batch_docids = None
                p_start = 0
                passage_batch_size = args.passage_batch_size

                while p_start < p_reps.shape[0]:
                    current_batch_size = min(passage_batch_size, p_reps.shape[0] - p_start)
                    p_batch = None
                    pointwise_scores = None
                    scores = None
                    indices = None
                    try:
                        p_batch = torch.from_numpy(
                            p_reps[p_start:p_start + current_batch_size]
                        ).to(device=device, dtype=score_dtype)
                        pointwise_scores = torch.einsum("qvd,pvd->qpv", q_batch, p_batch)
                        scores = torch.logsumexp(pointwise_scores.float() / args.logsumexp_temperature, dim=-1)
                        scores = scores * args.logsumexp_temperature
                        k = min(args.depth, scores.size(1))
                        scores, indices = torch.topk(scores, k=k, dim=1)
                    except torch.cuda.OutOfMemoryError:
                        del p_batch, pointwise_scores, scores, indices
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        if current_batch_size == 1:
                            raise RuntimeError(
                                "Adder search OOM with passage_batch_size=1. Reduce --batch_size."
                            )
                        passage_batch_size = max(1, current_batch_size // 2)
                        logger.warning(
                            "CUDA OOM on %s; reducing passage_batch_size to %s",
                            device,
                            passage_batch_size,
                        )
                        continue

                    docids = p_lookup[indices.cpu().numpy() + p_start]
                    batch_scores, batch_docids = merge_topk(
                        batch_scores,
                        batch_docids,
                        scores.cpu().numpy(),
                        docids,
                        args.depth,
                    )
                    p_start += current_batch_size
                    del p_batch, pointwise_scores, scores, indices

                shard_scores.append(batch_scores)
                shard_docids.append(batch_docids)
                if progress_bar is not None:
                    progress_bar.update(1)

            worker_scores, worker_docids = merge_topk(
                worker_scores,
                worker_docids,
                np.concatenate(shard_scores, axis=0),
                np.concatenate(shard_docids, axis=0),
                args.depth,
            )
            del p_reps

    return worker_scores, worker_docids.astype(str)


def search_adder_shards(q_reps, q_lookup, index_files, args):
    devices = args.devices.split(',') if args.devices else [args.device]
    devices = [device.strip() for device in devices if device.strip()]
    if not devices:
        devices = ['cuda' if torch.cuda.is_available() else 'cpu']

    if len(devices) == 1:
        return score_adder_shards_on_device(q_reps, index_files, args, devices[0], show_progress=True)

    shard_groups = [index_files[i::len(devices)] for i in range(len(devices))]
    worker_scores = None
    worker_docids = None
    query_batches = (q_reps.shape[0] + args.batch_size - 1) // args.batch_size
    total_work_units = sum(len(shard_group) for shard_group in shard_groups) * query_batches
    with tqdm(total=total_work_units, desc='Scoring shard/query batches', disable=args.quiet) as progress_bar:
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [
                executor.submit(
                    score_adder_shards_on_device,
                    q_reps,
                    shard_group,
                    args,
                    device,
                    False,
                    progress_bar,
                )
                for device, shard_group in zip(devices, shard_groups)
                if shard_group
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc='Merging GPU top-k', disable=args.quiet):
                scores, docids = future.result()
                worker_scores, worker_docids = merge_topk(
                    worker_scores,
                    worker_docids,
                    scores,
                    docids,
                    args.depth,
                )
    return worker_scores, worker_docids.astype(str)


def write_ranking(corpus_indices, corpus_scores, q_lookup, ranking_save_file):
    with open(ranking_save_file, 'w') as f:
        for qid, q_doc_scores, q_doc_indices in zip(q_lookup, corpus_scores, corpus_indices):
            score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
            for s, idx in score_list:
                f.write(f'{qid}\t{idx}\t{s}\n')


def pickle_load(path):
    with open(path, 'rb') as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup


def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def main():
    parser = ArgumentParser()
    parser.add_argument('--query_reps', required=True)
    parser.add_argument('--passage_reps', required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--depth', type=int, default=1000)
    parser.add_argument('--save_ranking_to', required=True)
    parser.add_argument('--save_text', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--score_function', choices=['auto', 'dot', 'adder'], default='auto')
    parser.add_argument('--passage_batch_size', type=int, default=4096)
    parser.add_argument('--logsumexp_temperature', type=float, default=1.0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--devices', default=None, help='Comma-separated devices for Adder shard-parallel search.')
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='float32')

    args = parser.parse_args()

    index_files = glob.glob(args.passage_reps)
    logger.info(f'Pattern match found {len(index_files)} files; loading them into index.')

    p_reps_0, p_lookup_0 = pickle_load(index_files[0])
    q_reps, q_lookup = pickle_load(args.query_reps)

    score_function = args.score_function
    if score_function == 'auto':
        score_function = 'adder' if q_reps.ndim == 3 or p_reps_0.ndim == 3 else 'dot'

    if score_function == 'adder':
        logger.info('Adder Search Start')
        all_scores, psg_indices = search_adder_shards(q_reps, q_lookup, index_files, args)
        logger.info('Adder Search Finished')
        if args.save_text:
            write_ranking(psg_indices, all_scores, q_lookup, args.save_ranking_to)
        else:
            pickle_save((all_scores, psg_indices), args.save_ranking_to)
        return

    retriever = FaissFlatSearcher(p_reps_0)

    shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
    if len(index_files) > 1:
        shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))
    look_up = []
    for p_reps, p_lookup in shards:
        retriever.add(p_reps)
        look_up += p_lookup

    num_gpus = faiss.get_num_gpus()
    if num_gpus == 0:
        logger.info("No GPU found or using faiss-cpu. Back to CPU.")
    else:
        logger.info(f"Using {num_gpus} GPU")
        if num_gpus == 1:
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            res = faiss.StandardGpuResources()
            retriever.index = faiss.index_cpu_to_gpu(res, 0, retriever.index, co)
        else:
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True
            retriever.index = faiss.index_cpu_to_all_gpus(retriever.index, co,
                                                     ngpu=num_gpus)

    logger.info('Index Search Start')
    all_scores, psg_indices = search_queries(retriever, q_reps, look_up, args)
    logger.info('Index Search Finished')

    if args.save_text:
        write_ranking(psg_indices, all_scores, q_lookup, args.save_ranking_to)
    else:
        pickle_save((all_scores, psg_indices), args.save_ranking_to)


if __name__ == '__main__':
    main()
