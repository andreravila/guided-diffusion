"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import argparse
import os
import time

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
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from guided_diffusion.train_util import parse_resume_step_from_filename

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
    resume_step = parse_resume_step_from_filename(args.model_path)
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading data...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.large_size,
        class_cond=args.class_cond,
        deterministic=True,
        num_samples=args.num_samples
    )
    # As it is deterministic, we know the indexes of the samples loaded,
    # because they are loaded in order
    indexes = []
    for entry in sorted(bf.listdir(args.data_dir)):
        entry = entry.replace(".png", "")
        indexes.append(entry)

    os.makedirs(args.out_dir, exist_ok=True)

    logger.log("creating samples...")
    logger.log(f"saving to {args.out_dir}")
    s = 0
    avg_psnr = 0
    avg_ssim = 0

    start_time = time.time()
    while s < args.num_samples:

        high_res, model_kwargs = next(data)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        sample = diffusion.p_sample_loop(
            model,
            (args.batch_size, 1, args.large_size, args.large_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        for i in range(0, high_res.size(0)):
            index = indexes[ s + i]

            out_path = os.path.join(args.out_dir, f"{resume_step}_{index}_hr.png")
            high_res_array = high_res[i][0].numpy()
            high_res_array = ((high_res_array + 1) * 127.5).clip(0, 255).astype(np.uint8)
            Image.fromarray(high_res_array).save(out_path)

            out_path = os.path.join(args.out_dir, f"{resume_step}_{index}_lr.png")
            low_res_array = model_kwargs['low_res'][i][0].cpu().numpy()
            low_res_array = ((low_res_array + 1) * 127.5).clip(0, 255).astype(np.uint8)
            Image.fromarray(high_res_array).save(out_path)

            out_path = os.path.join(args.out_dir, f"{resume_step}_{index}_sr.png")
            super_res_array = sample[i][0].cpu().numpy()
            super_res_array = ((super_res_array + 1) * 127.5).clip(0, 255).astype(np.uint8)
            Image.fromarray(super_res_array).save(out_path)

            avg_psnr += peak_signal_noise_ratio(high_res_array, super_res_array)
            avg_ssim += structural_similarity(high_res_array, super_res_array, data_range=255)

        logger.log(f"created {str(s + high_res.size(0))} samples")
        s += args.batch_size

    avg_psnr /= args.num_samples
    avg_ssim /= args.num_samples
    total_time = int(time.time() - start_time)

    logger.log(f"Step {resume_step}:")
    logger.log(f"Average PSNR: {avg_psnr}, Average SSIM: {avg_ssim}, Total sampling time: {str(total_time)} seconds")


    with open(os.path.join(args.out_dir, "sampling.txt"),"a") as file:
        file.write(f"Step {resume_step}:\n")
        file.write(f"Average PSNR: {avg_psnr}, Average SSIM: {avg_ssim}, Total sampling time: {str(total_time)} seconds")

    dist.barrier()
    logger.log("sampling complete")



def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=3,
        batch_size=3,
        use_ddim=False,
        data_dir="./dataset-final/slices-dataset-png/train/hr_128",
        model_path="./model000300.pt",
        out_dir="./dataset-final/slices-dataset-png/output"
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
