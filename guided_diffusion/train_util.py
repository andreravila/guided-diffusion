import copy
import functools
import os

import time
import blobfile as bf
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        val_data=None,
        val_indexes=None,
        val_interval=1000,
        val_save_suffix=None,
        val_out_dir=None,
        just_validate=False,
        image_size=None,
        clip_denoised=False,
    ):
        self.val_interval = val_interval
        self.val_data = val_data
        self.val_indexes = val_indexes
        self.val_save_suffix = val_save_suffix
        self.val_out_dir = val_out_dir
        self.just_validate = just_validate
        self.image_size = image_size
        self.clip_denoised = clip_denoised

        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            if self.step % self.val_interval == 0:
                self.validate()    
            
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


    def validate(self):
        os.makedirs(self.val_out_dir, exist_ok=True)

        logger.log("creating samples...")
        logger.log(f"saving to {self.val_out_dir}")
        s = 0
        avg_psnr = 0
        avg_ssim = 0
        start_time = time.time()
        batch_size = self.batch_size 

        # The len of val_indexes is the number of samples is the number of items in the validation
        # directory if num_samples is not specified, otherwise it is equal to num_samples
        val_num_samples = len(self.val_indexes)

        while s < val_num_samples:

            high_res, model_kwargs = next(self.val_data)
            model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}

            if val_num_samples - s < batch_size:
                batch_size = val_num_samples - s
                model_kwargs['low_res'] = model_kwargs['low_res'][:batch_size]

            sample = self.diffusion.p_sample_loop(
                self.model,
                (batch_size, 1, self.image_size, self.image_size),
                clip_denoised=self.clip_denoised,
                model_kwargs=model_kwargs,
            )

            for i in range(0, batch_size):
                index = self.val_indexes[ s + i]

                # When just validating, we only want to save the output of the model,
                # which is the super-resolution image
                if self.just_validate == True:
                    out_path_sr = os.path.join(self.val_out_dir, f"{index}_sr.{self.val_save_suffix}")
                    super_res_array = sample[i][0].cpu().numpy()
                    high_res_array = high_res[i][0].numpy()

                    if self.val_save_suffix == "npy":
                        super_res_array = transform_and_save_npy(super_res_array, out_path_sr)
                        high_res_array = (high_res_array + 1) / 2
                        data_range=1
                    else:
                        super_res_array = transform_and_save_image(super_res_array, out_path_sr)
                        high_res_array = ((high_res_array + 1) * 127.5).clip(0, 255).astype(np.uint8)
                        data_range=255

                else:
                    out_path_hr = os.path.join(self.val_out_dir, f"{self.step}_{index}_hr.{self.val_save_suffix}")
                    high_res_array = high_res[i][0].numpy()

                    out_path_lr = os.path.join(self.val_out_dir, f"{self.step}_{index}_lr.{self.val_save_suffix}")
                    low_res_array = model_kwargs['low_res'][i][0].cpu().numpy()

                    out_path_sr = os.path.join(self.val_out_dir, f"{self.step}_{index}_sr.{self.val_save_suffix}")
                    super_res_array = sample[i][0].cpu().numpy()

                    if self.val_save_suffix == "npy":
                        high_res_array = transform_and_save_npy(high_res_array, out_path_hr)
                        low_res_array =  transform_and_save_npy(low_res_array, out_path_lr)
                        super_res_array = transform_and_save_npy(super_res_array, out_path_sr)
                        data_range=1
                    else:
                        high_res_array = transform_and_save_image(high_res_array, out_path_hr)
                        low_res_array = transform_and_save_image(low_res_array, out_path_lr)
                        super_res_array = transform_and_save_image(super_res_array, out_path_sr)
                        data_range=255

                avg_psnr += peak_signal_noise_ratio(high_res_array, super_res_array)
                avg_ssim += structural_similarity(high_res_array, super_res_array, data_range=data_range)

            logger.log(f"created {str(s + batch_size)} samples")
            s += batch_size

        avg_psnr /= val_num_samples
        avg_ssim /= val_num_samples
        total_time = int(time.time() - start_time)

        logger.log(f"Step {self.step}:")
        logger.log(f"Average PSNR: {avg_psnr}, Average SSIM: {avg_ssim}, Total sampling time: {str(total_time)} seconds")


        with open(os.path.join(self.val_out_dir, "sampling.txt"),"a") as file:
            file.write(f"Step {self.step}:\n")
            file.write(f"Average PSNR: {avg_psnr}, Average SSIM: {avg_ssim}, Total sampling time: {str(total_time)} seconds")

        dist.barrier()
        logger.log("sampling complete")

def transform_and_save_image(sample_array, path):
    sample_array = ((sample_array + 1) * 127.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(sample_array).save(path)
    return sample_array

def transform_and_save_npy(sample_array, path):
    sample_array = (sample_array + 1) / 2
    np.save(path, sample_array)
    return sample_array


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
