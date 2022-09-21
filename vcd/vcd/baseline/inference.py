#!/usr/bin/env python3
import argparse
import enum
import logging
import os

from torch import multiprocessing


class InferenceTransforms(enum.Enum):
    SSCD = enum.auto()


class Accelerator(enum.Enum):
    CPU = enum.auto()
    CUDA = enum.auto()


parser = argparse.ArgumentParser()
inference_parser = parser.add_argument_group("Inference")
inference_parser.add_argument("--torchscript_path", required=True)
inference_parser.add_argument("--batch_size", type=int, default=32)
inference_parser.add_argument("--distributed_rank", type=int, default=0)
inference_parser.add_argument("--distributed_size", type=int, default=1)
inference_parser.add_argument("--processes", type=int, default=1)
inference_parser.add_argument(
    "--transforms", choices=[x.name for x in InferenceTransforms], default="SSCD"
)
inference_parser.add_argument(
    "--accelerator", choices=[x.name.lower() for x in Accelerator], default="cpu"
)
inference_parser.add_argument("--output_path", required=True)

dataset_parser = parser.add_argument_group("Dataset")
dataset_parser.add_argument("--dataset_path", required=True)
dataset_parser.add_argument("--fps", default=1, type=float)
dataset_parser.add_argument("--video_extensions", default="mp4")
dataset_parser.add_argument("--ffmpeg_path", default="ffmpeg")


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inference.py")
logger.setLevel(logging.INFO)


def main(args):
    success = False
    if args.processes > 1 and args.distributed_size > 1:
        raise Exception(
            "Set either --processes (single-machine distributed) or "
            "both --distributed_size and --distributed_rank (arbitrary "
            "distributed)"
        )
    if args.processes > 1:
        processes = []
        logger.info(f"Spawning {args.processes} processes")
        accelerator = Accelerator[args.accelerator.upper()]
        backend = "nccl" if accelerator == Accelerator.CUDA else "gloo"
        multiprocessing.set_start_method("spawn")
        try:
            for rank in range(args.processes):
                p = multiprocessing.Process(
                    target=distributed_worker_process,
                    args=(args, rank, args.processes, backend),
                )
                processes.append(p)
                p.start()
            worker_success = []
            for p in processes:
                p.join()
                worker_success.append(p.exitcode == os.EX_OK)
            success = all(worker_success)
        finally:
            for p in processes:
                p.kill()
    else:
        worker_process(args, args.distributed_rank, args.distributed_size)
        success = True

    if success:
        logger.info("Inference succeeded.")
    else:
        logger.error("Inference FAILED!")


def distributed_worker_process(args, rank, world_size, backend):
    from torch import distributed

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "19529"
    distributed.init_process_group(backend, rank=rank, world_size=world_size)
    worker_process(args, rank, world_size)


def worker_process(args, rank, world_size):
    # Late import: initialize cuda after worker spawn.
    from vcd.baseline.inference_impl import worker_process as worker_impl

    return worker_impl(args, rank, world_size)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
