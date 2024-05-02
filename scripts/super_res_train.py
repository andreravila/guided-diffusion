"""
Train a super-resolution model.
"""

import argparse
import os

import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion import image_datasets
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.large_size,
        class_cond=args.class_cond,
    )

    val_data = load_data(
        data_dir=args.val_data_dir,
        batch_size=args.batch_size,
        image_size=args.large_size,
        class_cond=args.class_cond,
        deterministic=True,
        num_samples=args.val_num_samples
    )
    
    if val_data is not None:
        # As it is deterministic, we know the indexes of the samples loaded,
        # because they are loaded in order
        val_indexes =  image_datasets._list_image_files_recursively(args.val_data_dir, args.val_num_samples)
        for i in range(len(val_indexes)):
            val_indexes[i] = os.path.basename(val_indexes[i])
            val_indexes[i], _ = os.path.splitext(val_indexes[i])

    logger.log("training...")
    trainLoop = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,

        val_data=val_data,
        val_indexes=val_indexes,
        val_interval=args.val_interval,
        val_save_suffix=args.val_save_suffix,
        just_validate=args.just_validate,
        image_size=args.large_size,
        clip_denoised=args.clip_denoised,
        val_out_dir=args.val_out_dir
    )

    if args.just_validate == True:
        trainLoop.validate()
    else:
        trainLoop.run_loop()


def load_superres_data(data_dir, batch_size, large_size, small_size, class_cond=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=large_size,
        class_cond=class_cond,
        random_flip=False
    )
    for large_batch, model_kwargs in data:
        #model_kwargs["low_res"] = F.interpolate(large_batch, small_size, mode="area")
        yield large_batch, model_kwargs


def create_argparser():
    defaults = dict(
        data_dir="./dataset-final/slices-dataset-png/train/hr_128",
        val_data_dir="./dataset-final/slices-dataset-png/validate/hr_128",
        val_out_dir="./dataset-final/slices-dataset-png/val-output",
        just_validate = False,
        val_save_suffix = "png",
        val_num_samples=None,
        clip_denoised=True,
        schedule_sampler="uniform",
        lr=1e-5,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=2,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=100,
        val_interval=200,
        resume_checkpoint="model000300.pt",
        use_fp16=False,
        fp16_scale_growth=1e-3
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
