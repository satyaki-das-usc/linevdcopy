import functools
import os
from argparse import ArgumentParser
from multiprocessing import Manager, Pool, Queue, cpu_count

import pandas as pd
from ray import tune
from tqdm import tqdm

import sastvd as svd
import sastvd.helpers.joern as svdj
import sastvd.linevd.run as lvdrun

USE_CPU = cpu_count()


def process_joern_parallel(joern_input, queue: Queue):
    func_str, dataset, iid = joern_input
    svdj.full_run_joern_from_string(func_str, dataset, iid)
    return iid


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        required=False,
        help="The input training data file (a csv file).",
    )

    args, unknown = parser.parse_known_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    savedir = svd.get_dir(svd.cache_dir() / "minimal_datasets")
    df = pd.read_parquet(
        savedir / f"minimal_bigvul_False.pq", engine="fastparquet"
    ).dropna()

    funcs = df["before"].tolist()
    datasets = df["dataset"].tolist()
    iids = df["id"].tolist()

    if args.limit:
        funcs = funcs[: args.limit]
        datasets = datasets[: args.limit]
        iids = iids[: args.limit]

    func_cnt = len(funcs)

    joern_inputs = [
        (func_str, dataset, iid)
        for func_str, dataset, iid in zip(funcs, datasets, iids)
    ]

    with Manager() as m:
        message_queue = m.Queue()
        pool = Pool(USE_CPU)

        process_func = functools.partial(process_joern_parallel, queue=message_queue)

        done_iids = [
            iid
            for iid in tqdm(
                pool.imap_unordered(process_func, joern_inputs),
                desc=f"Functions",
                total=func_cnt,
            )
        ]

        message_queue.put("finished")
        pool.close()
        pool.join()
