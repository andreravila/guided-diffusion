"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

from guided_diffusion.image_datasets import load_data
import time
from PIL import Image

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading data...")
    #data = load_data_for_worker(args.base_samples, args.batch_size, args.class_cond)

    override_samples = False

    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.large_size,
        class_cond=args.class_cond,
        deterministic=True,
        use_fp16=args.use_fp16,
        num_samples=args.num_samples,
        out_dir=args.out_dir,
        override_samples=override_samples
    )

    logger.log(f"Output dir: {args.out_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    # As it is deterministic, we know the indexes of the samples loaded,
    # because they are loaded in order
    indexes = []

    # Don't sample again the samples if they are already saved
    if override_samples == False:
        sampled_files = sorted(bf.listdir(args.out_dir))

    for entry in sorted(bf.listdir(args.data_dir)):
        if override_samples == True or entry not in sampled_files:
            entry = entry.split("/")[-1].split(".")[0]
            indexes.append(entry)

    logger.log("creating samples...")

    start_time = time.time()
    i = 0
    if args.num_samples is None:
        args.num_samples = len(indexes)
        
    while i * args.batch_size < args.num_samples:
        high_res, model_kwargs  = next(data)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}

        in_channels = model_kwargs["low_res"].shape[1]

        sample_batch= diffusion.p_sample_loop(
            model,
            (args.batch_size, in_channels, args.large_size, args.large_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        if args.save_suffix == "npy":
            sample_batch= ((sample_batch+ 1) / 2).clamp(0, 1).to(th.uint8)
        else:
            sample_batch= ((sample_batch+ 1) * 127.5).clamp(0, 255).to(th.uint8)
        
        sample_batch= sample_batch.permute(0, 2, 3, 1)
        sample_batch= sample_batch.contiguous()

        # Commented lines are for multi-gpu sampling
        #all_sample_batches = [th.zeros_like(sample_batch) for _ in range(dist.get_world_size())]
        #dist.all_gather(all_sample_batches, sample_batch)  # gather not supported with NCCL

        #for sample_batch in all_sample_batches:

        index_offset = i * args.batch_size  # Calculate the starting index for the current batch
        sample_array = sample_batch.cpu().numpy()
        for j in range(args.batch_size):
            sample = sample_array[j]
            sample = sample.squeeze()  # Remove extra dimension if needed
            idx = indexes[index_offset + j]  # Get the corresponding index for the sample
            path = os.path.join(args.out_dir, f"{idx}.{args.save_suffix}")
            if args.save_suffix == "npy":
                np.save(path, sample)
            else:
                Image.fromarray(sample).save(path)

        i += 1
        logger.log(f"saved {i * args.batch_size} samples")
        logger.log(f"Time: {time.time() - start_time:.06}s")
        start_time = time.time()

    logger.log("sampling complete")


def load_data_for_worker(base_samples, batch_size, class_cond):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]
        if class_cond:
            label_arr = obj["arr_1"]
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    buffer = []
    label_buffer = []
    while True:
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i])
            if class_cond:
                label_buffer.append(label_arr[i])
            if len(buffer) == batch_size:
                batch = th.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                res = dict(low_res=batch)
                if class_cond:
                    res["y"] = th.from_numpy(np.stack(label_buffer))
                yield res
                buffer, label_buffer = [], []


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=None,
        save_suffix = "npy",
        batch_size=16,
        use_ddim=False,
        use_fp16=True,
        # Pass */hr_128 as path, when loading the dataset it will load the hig_res path, replace it with sr_16_128, and load the low_res path
        data_dir="./dataset3TSubsetSliced/sliced_dataset_dki_mppca_144_05/test/hr_128",
        out_dir="./dataset3TSubsetSliced/sliced_dataset_dki_mppca_144_05/estimated_samples",
        model_path="checkpoint_model/checkpoint_dki_mppca_144_05/model/model100000.pt",
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
