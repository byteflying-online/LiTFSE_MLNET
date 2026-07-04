# python 3
# -*- coding:utf-8 -*-
"""
Created on 2026/04/17
@author: Simon Li
"""

"""
Set <--data_dir> to the directory of the MCC5_THU_Motor dataset firstly! (for example: 'D:/data/MCC5_THU_Motor')
"""
import os
import time
import torch
import random
import logging
import argparse
import numpy as np
from datetime import datetime
from utils.logger import setlogger
from utils.train_utils import train_utils

# Random setting
seed = 88
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

parse_bool = lambda x: x.lower() in ["yes", "true", "t", "y", "1"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="D:/data/MCC5_THU_Motor",
        help="the directory of the dataset",
    )
    parser.add_argument(
        "--data_name", type=str, default="THU_Motor_speed", choices=["THU_Motor_speed"], help="the name of the dataset"
    )
    parser.add_argument(
        "--data_mode",
        type=str,
        default="transient_robust",
        choices=["transient_robust", "steady"],
        help="the mode of the dataset: steady/transient_robust",
    )
    parser.add_argument("--data_type", type=str, default="time", choices=["time"], help="the data_type of the dataset")
    parser.add_argument(
        "--normlizetype",
        type=str,
        choices=["0-1", "-1-1", "mean-std"],
        default="mean-std",
        help="data normalization methods",
    )
    parser.add_argument("--data_signalsize", type=int, default=8192, help="length of each signal segment")
    parser.add_argument(
        "--overlap_ratio",
        type=float,
        default=0.75,
        help="Sliding window overlap ratio. 0.5=50%% overlap, 0.75=75%% overlap, 0=no overlap",
    )
    parser.add_argument("--SNR", type=float, default=40, help="activate when SNR in (-100,100) else set to None")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size of the training process")
    parser.add_argument("--num_workers", type=int, default=2, help="the number of dataloader workers")
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of dataset to include in test split (e.g., 0.3 for 30% test)",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Proportion of dataset to include in validation split (e.g., 0.3 for 30% validation)",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="LiTFSE_MLNet",
        choices=[
            "LiTFSE_MLNet",
            "LiConvFormer",
            "ResNet18",
            "TFN_STTF",
            "TFN_Chirplet",
            "TFN_Morlet",
            "Backbone_CNN",
            "Random_CNN",
        ],
        help="the model to be trained",
    )
    parser.add_argument("--kernel_size", type=int, default=11, help="the kernel size of traditional conv layer")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=r"./checkpoint",
        help="Directory to save model checkpoints, training logs, and evaluation results",
    )

    # func-models parameters
    parser.add_argument(
        "--mid_channel",
        type=int,
        default=32,
        choices=[16, 32, 64, 128],
        help="the channel number of preprocessing layer",
    )
    parser.add_argument(
        "--clamp_flag", type=parse_bool, default="True", help="flag to limit the superparams of TFconv layer"
    )
    parser.add_argument(
        "--freq_init_type",
        type=str,
        default="uniform",
        choices=["linear", "log", "uniform"],
        help="initialization type for TFconv frequencies",
    )

    parser.add_argument(
        "--emb_size",
        type=int,
        default=256,
        choices=[64, 128, 256],
        help="STFormer: patch embedding length (n_patch=T/emb_size)",
    )
    parser.add_argument("--n_head", type=int, default=2, choices=[2, 4, 8], help="STFormer: number of attention heads")
    parser.add_argument(
        "--n_layer", type=int, default=1, choices=[1, 2, 3, 4], help="STFormer: number of transformer layers"
    )

    #   --use_freqse 1111  : All four layers use FreqSE (full model, default)
    #   --use_freqse 0000  : No layer uses FreqSE (core ablation: verifying FreqSE overall contribution)
    #   --use_freqse 1000  : Only conv1 uses FreqSE (layer-level ablation: verifying shallow FreqSE contribution)
    #   --use_freqse 0110  : Only conv2/conv3 use FreqSE (layer-level ablation: verifying deep FreqSE contribution)
    # Applicable model: only effective for LiTFSE_MLNet; other models ignore this parameter
    parser.add_argument(
        "--use_freqse",
        type=str,
        default="1111",
        help="FreqSE ablation: 4-digit string controlling FreqSE at conv1-conv4 (1=use, 0=skip). Default: 1111 (full model)",
    )

    # optimization information
    parser.add_argument("--opt", type=str, choices=["sgd", "adam", "RMSprop"], default="adam", help="the optimizer")
    parser.add_argument("--lr", type=float, default=1e-3, help="the initial learning rate")
    parser.add_argument("--patience", type=int, default=10, help="the para of lr scheduler")
    parser.add_argument("--momentum", type=float, default=0.9, help="the momentum for sgd")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 penalty) for optimizer. Typical values: 0, 1e-5, 1e-4, 1e-3",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        choices=["step", "exp", "stepLR", "fix"],
        default="stepLR",
        help='step: MultiStepLR at specified epochs (--steps "30,60"); '
        "stepLR: StepLR every fixed epochs (--steps 30); "
        "exp: ExponentialLR; fix: constant LR",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="learning rate scheduler parameter for step and exp")
    parser.add_argument(
        "--steps",
        type=str,
        default="30",
        help='For stepLR: integer interval (e.g., 30). For step: comma-separated epochs (e.g., "30,60,80")',
    )

    # save, load and display information
    parser.add_argument("--save_model", type=parse_bool, default="True", help="max number of epoch")
    parser.add_argument(
        "--save_best_model", type=parse_bool, default="True", help="save the best model according to the val accuracy"
    )
    parser.add_argument("--max_epoch", type=int, default=15, help="max number of epoch")
    parser.add_argument("--print_step", type=int, default=10, help="the interval of log training information")
    parser.add_argument(
        "--gen_data_only", type=parse_bool, default="False", help="if True, only preprocess and save dataset, then exit"
    )

    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    os.chdir(os.path.split(os.path.realpath(__file__))[0])

    args = parse_args()
    # Process the args.SNR
    if args.SNR <= -1e2 or args.SNR >= 1e2:
        args.SNR = None

    args.use_freqse = [int(c) for c in args.use_freqse]
    model_lower = args.model_name.lower()
    if "sttf" in model_lower:
        fmin, fmax = 0.03, 0.45
        param_dim = 1
    elif "chirplet" in model_lower:
        fmin, fmax = 0.03, 0.45
        param_dim = 1
    elif "morlet" in model_lower:
        fmin, fmax = 0.4, 10.0
        param_dim = 1
    else:
        fmin, fmax = 0.03, 0.45
        param_dim = 1

    if args.freq_init_type == "linear":
        init_freqs = np.linspace(fmin, fmax, args.mid_channel).astype(np.float32)
    elif args.freq_init_type == "log":
        log_fmin, log_fmax = np.log(fmin), np.log(fmax)
        log_vals = np.linspace(log_fmin, log_fmax, args.mid_channel)
        init_freqs = np.exp(log_vals).astype(np.float32)
    elif args.freq_init_type == "uniform":
        init_freqs = np.random.uniform(fmin, fmax, args.mid_channel).astype(np.float32)
    else:
        init_freqs = np.linspace(fmin, fmax, args.mid_channel).astype(np.float32)

    args.init_freqs = init_freqs

    if args.gen_data_only:
        from Datasets import THU_Motor  # THU_Motor

        dataset_args = {
            "data_dir": args.data_dir,
            "data_name": args.data_name,
            "normlizetype": args.normlizetype,
            "test_size": args.test_size,
            "val_size": args.val_size,
            "data_mode": args.data_mode,
            "data_signalsize": args.data_signalsize,
            "overlap_ratio": args.overlap_ratio,
        }
        thu = THU_Motor(dataset_args)
        (train_ds, val_ds, test_ds), label_name = thu.data_preprare(signal_size=args.data_signalsize, SNR=args.SNR)
        print("Dataset generated and saved, program exiting.")
        exit(0)

    # Prepare the saving path for the models
    sub_dir = (
        args.model_name
        + "-"
        + args.data_name
        + "-"
        + args.data_mode
        + "-"
        + args.data_type
        + "-"
        + datetime.strftime(datetime.now(), "%m%d-%H%M%S")
    )
    save_dir = os.path.join(args.checkpoint_dir, sub_dir).replace("\\", "/")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, "training.log"))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("<args> {}: {}".format(k, v))

    save_dir_sub = os.path.join(save_dir, "postprosess")
    if not os.path.exists(save_dir_sub):
        os.makedirs(save_dir_sub)

    def train_ds():
        # initialize the trainer
        trainer = train_utils(args, save_dir)
        trainer.setup()

        # train the model
        time_start_train = time.time()
        trainer.train()
        logging.info("<training time>: {:.3f}".format(time.time() - time_start_train))

        # plot the results
        trainer.plot_save()

    train_ds()
