"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

from guided_diffusion.image_datasets import load_data
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    sr_model_and_diffusion_defaults,
    sr_classifier_defaults,
    sr_create_model_and_diffusion,
    sr_create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)

import blobfile as bf
import time
from PIL import Image


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
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

    logger.log("loading classifier...")
    classifier = sr_create_classifier(**args_to_dict(args, sr_classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)
    
    
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
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample_batch = sample_fn(
            model_fn,
            (args.batch_size, 1, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )
        if args.save_suffix == "npy":
            sample_batch= ((sample_batch + 1) / 2).clamp(0, 1).to(th.uint8)
        else:
            sample_batch= ((sample_batch + 1) * 127.5).clamp(0, 255).to(th.uint8)
        
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


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    defaults.update(sr_model_and_diffusion_defaults())
    defaults.update(sr_classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
