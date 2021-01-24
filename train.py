import os
import time
import argparse
import math
import numpy as np
from numpy import finfo

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
from torch import autograd
import pytorch_warmup as warmup

from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams

from tqdm import tqdm


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams, use_textgrid=True)
    valset = TextMelLoader(hparams.validation_files, hparams, use_textgrid=True, dataset_type='val')
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=hparams.num_workers, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=True,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    model = Tacotron2(hparams).cuda()

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer, scaler, step_scheduler, warmup_scheduler):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    scaler.load_state_dict(checkpoint_dict['scaler'])
    step_scheduler.load_state_dict(checkpoint_dict['step_scheduler'])
    warmup_scheduler.load_state_dict(checkpoint_dict['warmup_scheduler'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, scaler, step_scheduler, warmup_scheduler, learning_rate, iteration


def save_checkpoint(model, optimizer, scaler, step_scheduler, warmup_scheduler, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'step_scheduler': step_scheduler.state_dict(),
                'warmup_scheduler': warmup_scheduler.state_dict(),
                'learning_rate': learning_rate}, filepath)


def teacher_forcing(model, dataset, batch_size, n_gpus, collate_fn, distributed_run):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        data_sampler = DistributedSampler(dataset) if distributed_run else None
        data_loader = DataLoader(dataset, sampler=data_sampler, num_workers=4,
                                shuffle=False, batch_size=32,
                                pin_memory=False, collate_fn=collate_fn)

        for i, batch in enumerate(tqdm(data_loader, position=0, leave=True), 1):
            x, y, audio_names = model.parse_batch(batch, teacher_forcing=True)
            output_lengths = x[-1]
            _, mel_out_postnet, _, _ = model(x)

            for j in range(len(x[2])):
                audio_name = ''.join([chr(idx) for idx in audio_names[j].tolist()])
                output_length = output_lengths[j].item()
                melspec = mel_out_postnet[j, :, :output_length]


                # TODO: This path is hardcoded and should not be
                np.save("LJSpeech/npy/"+audio_name+".npy", melspec.cpu().detach().numpy())


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=True, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss, losses_pkg = criterion(y_pred, y)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
                reduced_val_mel = reduce_tensor(losses_pkg[0].data, n_gpus).item()
                reduced_val_dur = reduce_tensor(losses_pkg[1].data, n_gpus).item()
                #reduced_val_tpse = reduce_tensor(losses_pkg[1].data, n_gpus).item()
                #reduced_val_dur = reduce_tensor(losses_pkg[2].data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
                reduced_val_mel = losses_pkg[0].item()
                reduced_val_dur = losses_pkg[1].item()
                #reduced_val_tpse = losses_pkg[1].item()
                #reduced_val_dur = losses_pkg[2].item()

            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
        logger.log_validation(val_loss, model, y, y_pred, iteration)


def train(args, output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)
    scaler = amp.GradScaler()
    step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.5)
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=4000)

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = Tacotron2Loss(hparams.lambda_duration)

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 1
    epoch_offset = 1
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, scaler, step_scheduler, warmup_scheduler, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer, scaler, step_scheduler, warmup_scheduler)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    if args.teacher_forcing:
        tf_dataset = TextMelLoader(hparams.validation_files, hparams, use_textgrid=True, dataset_type='all')
        teacher_forcing(model, tf_dataset, hparams.batch_size, n_gpus, collate_fn, hparams.distributed_run)

        return

    model.train()
    is_overflow = False

    # ================ MAIN TRAINING LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(tqdm(train_loader), 1):
            start = time.perf_counter()

            optimizer.zero_grad()

            with amp.autocast():
                x, y = model.parse_batch(batch)
                if args.profiling:
                    with torch.autograd.profiler.profile(use_cuda=True) as prof:
                        y_pred = model(x)
                    print(prof.key_averages().table(sort_by="self_cuda_time_total"))
                else:
                    y_pred = model(x)
                loss, losses_pkg = criterion(y_pred, y)

            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
                reduced_mel = reduce_tensor(losses_pkg[0].data, n_gpus).item()
                reduced_dur = reduce_tensor(losses_pkg[1].data, n_gpus).item()
                #reduced_tpse = reduce_tensor(losses_pkg[1].data, n_gpus).item()
                #reduced_dur = reduce_tensor(losses_pkg[2].data, n_gpus).item()
            else:
                reduced_loss = loss.item()
                reduced_mel = losses_pkg[0].item()
                reduced_dur = losses_pkg[1].item()
                #reduced_tpse = losses_pkg[1].item()
                #reduced_dur = losses_pkg[2].item()

            losses_pkg = reduced_mel, reduced_dur
            #losses_pkg = reduced_mel, reduced_tpse, reduced_dur

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)
            scaler.step(optimizer)
            scaler.update()
            step_scheduler.step(step_scheduler.last_epoch+1)
            warmup_scheduler.dampen()

            if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
                validate(model, criterion, valset, iteration,
                         hparams.batch_size, n_gpus, collate_fn, logger,
                         hparams.distributed_run, rank)
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, scaler, step_scheduler, warmup_scheduler, learning_rate, iteration,
                                    checkpoint_path)

            iteration += 1

            if not is_overflow and rank == 0:
                learning_rate = optimizer.param_groups[0]['lr']
                duration = time.perf_counter() - start
                print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration - 1, reduced_loss, grad_norm, duration))
                logger.log_training(
                    reduced_loss, grad_norm, learning_rate, duration, iteration - 1, losses_pkg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--teacher_forcing', action='store_true',
                        help='generate numpy files from teacher forcing')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    parser.add_argument('--profiling', action='store_true',
                        required=False, help='enables the profiler')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train(args, args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
