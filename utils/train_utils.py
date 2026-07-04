import os
import time
import torch
import Models
import random
import logging
import warnings
import Datasets
import numpy as np
from torch import nn
import scipy.io as io
import seaborn as sns
from torch import optim
import matplotlib.pyplot as plt
from utils.summary import summary

# set random seed
seed = 88
np.random.seed(seed)
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class train_utils(object):
    def __init__(self, args, save_dir: str):
        self.args = args
        self.save_dir = save_dir

    def setup(self):
        """
        Initialize the datasets, models, loss and optimizer
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = 1
            self.device_count = torch.cuda.device_count()
            logging.info("using {} gpus".format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info("using {} cpu".format(self.device_count))

        dataset = getattr(Datasets, "THU_Motor")

        self.datasets = {}
        subargs = {
            k: getattr(args, k)
            for k in [
                "data_dir",
                "data_type",
                "normlizetype",
                "test_size",
                "val_size",
                "data_mode",
                "data_signalsize",
                "overlap_ratio",
            ]
        }
        (self.datasets["train"], self.datasets["val"], self.datasets["test"]), self.label_name = dataset(
            subargs
        ).data_preprare(signal_size=args.data_signalsize, SNR=self.args.SNR)
        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.datasets[x],
                batch_size=args.batch_size,
                shuffle=(True if x == "train" else False),
                num_workers=args.num_workers,
                pin_memory=(True if self.device == "cuda" else False),
            )
            for x in ["train", "val"]
        }

        if self.datasets.get("test") is not None:
            self.dataloaders["test"] = torch.utils.data.DataLoader(
                self.datasets["test"],
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(self.device == "cuda"),
            )

        logging.info(f"dataset_train:{len(self.datasets['train']):d}, dataset_val:{len(self.datasets['val']):d}")
        logging.info(f"dataset_test:{len(self.datasets['test']):d}")

        logging.info(
            f"model_name: {args.model_name}, in_channels: {dataset.inputchannel}, out_channels: {dataset.num_classes}, kernel_size: {args.kernel_size}, clamp_flag: {args.clamp_flag}, mid_channel: {args.mid_channel}"
        )

        # Define the models
        self.model = getattr(Models, args.model_name)(
            in_channels=dataset.inputchannel,
            out_channels=dataset.num_classes,
            kernel_size=args.kernel_size,
            clamp_flag=args.clamp_flag,
            mid_channel=args.mid_channel,
            init_freqs=args.init_freqs,
            emb_size=args.emb_size,
            n_head=args.n_head,
            n_layer=args.n_layer,
            data_signalsize=args.data_signalsize,
            use_freqse=args.use_freqse,
        )

        self.model.to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"total_params: {total_params}, trainable_params: {trainable_params}")

        print(f"summary total: {total_params}, summary trainable: {trainable_params}")
        print(
            f"direct total: {sum(p.numel() for p in self.model.parameters())}, direct trainable: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

        # summary the model and record model info
        try:
            info = summary(self.model, self.datasets["train"][0][0].shape, batch_size=-1, device="cuda")
            for item in info.split("\n"):
                logging.info(item)
        except:
            print("summary does not work!")

        self.criterion = nn.BCEWithLogitsLoss()

        # Define the optimizer
        if args.opt == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
            )
        elif args.opt == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.opt == "RMSprop":
            self.optimizer = optim.RMSprop(
                self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
            )
        else:
            raise Exception("optimizer not implement")

        # Define the learning rate decay
        if args.lr_scheduler == "step":
            steps = [int(step) for step in args.steps.split(",")]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == "exp":
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == "stepLR":
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == "fix":
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

    def train(self):
        """
        Training process
        """

        args = self.args
        best_acc = 0.0
        step_start = time.time()
        early_stop_counter = 0
        patience = getattr(args, "patience", 10)
        stopped_early = False
        self.Records = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "best_epoch": 0}
        self.MinorRecords = {"train_loss": [], "train_acc": []}

        # Train the models via epochs
        for epoch in range(args.max_epoch):
            logging.info("-" * 5 + "Epoch {}/{}".format(epoch, args.max_epoch - 1) + "-" * 5)
            # Record the learning rate
            if self.lr_scheduler is not None:
                logging.info("current lr: {}".format(self.lr_scheduler.get_last_lr()))
            else:
                logging.info("current lr: {}".format(args.lr))

            # Each epoch has a training and val phase
            for phase in ["train", "val"]:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0
                batch_acc = 0
                batch_loss = 0.0
                batch_count = 0

                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()

                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    with torch.set_grad_enabled(phase == "train"):
                        logits = self.model(inputs)
                        if torch.isnan(logits).any() or torch.isinf(logits).any():
                            if phase == "train":
                                logging.warning(f"NaN/Inf in logits at batch {batch_idx}, skipping batch")
                                continue
                            else:
                                logging.warning(f"NaN/Inf in logits at val batch {batch_idx}")
                                break
                        loss = self.criterion(logits, labels)
                        pred = (torch.sigmoid(logits) > 0.5).float()
                        correct = (pred == labels).all(dim=1).float().sum().item()
                        loss_temp = loss.item() * inputs.size(0)
                        epoch_loss += loss_temp
                        epoch_acc += correct

                        # Calculate the training information
                        if phase == "train":
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += inputs.size(0)

                            # Print the training information
                            if (batch_idx + 1) % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_acc = batch_acc / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step
                                sample_per_sec = 1.0 * batch_count / train_time
                                logging.info(
                                    "<tempinfo> Epoch: {:3d} [{:4d}/{:4d}],  step: {:3d},  Train Loss: {:.3f},  Train Acc: {:.2f},  {:8.1f} samples/sec,  {:.4f} sec/batch".format(
                                        epoch,
                                        (batch_idx + 1) * len(inputs),
                                        len(self.dataloaders[phase].dataset),
                                        batch_idx + 1,
                                        batch_loss,
                                        batch_acc * 100,
                                        sample_per_sec,
                                        batch_time,
                                    )
                                )
                                self.MinorRecords["train_loss"].append(batch_loss)
                                self.MinorRecords["train_acc"].append(batch_acc)
                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0

                epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = epoch_acc / len(self.dataloaders[phase].dataset)
                logging.info(
                    "<info> Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.4f} sec".format(
                        epoch, phase, epoch_loss, phase, epoch_acc * 100, time.time() - epoch_start
                    )
                )

                # Record the epoch information
                self.Records["%s_loss" % phase].append(epoch_loss)
                self.Records["%s_acc" % phase].append(epoch_acc)

                # save the best and the final model if needed
                if phase == "val":
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        self.Records["best_epoch"] = epoch
                        logging.info("save best epoch {}, best acc {:.4f}".format(epoch, epoch_acc))
                        save_best_data_dir = os.path.join(
                            self.save_dir, "epoch{}-acc{:.4f}-best_model.pth".format(epoch, epoch_acc * 100)
                        )
                        save_best_data = {
                            "epoch": epoch + 1,
                            "state_dict": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "opt": self.args.opt,
                        }
                        if args.save_best_model:
                            torch.save(save_best_data, save_best_data_dir)
                            logging.info(f"Best model saved to {save_best_data_dir}")
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                        logging.info(f"No improvement on val acc for {early_stop_counter} epoch(s)")
                        if early_stop_counter >= patience:
                            logging.info(f"Early stopping at epoch {epoch}")
                            stopped_early = True
                            break

                    if epoch == args.max_epoch - 1:
                        if args.save_model:
                            # save the final models
                            logging.info("save final epoch {}, final acc {:.4f}".format(epoch, epoch_acc))
                            save_data = {
                                "epoch": epoch + 1,
                                "state_dict": self.model.state_dict(),
                                "optimizer": self.optimizer.state_dict(),
                                "opt": self.args.opt,
                            }
                            torch.save(
                                save_data,
                                os.path.join(
                                    self.save_dir, "epoch{}-acc{:.4f}-final_model.pth".format(epoch, epoch_acc * 100)
                                ),
                            )

            # Update the learning rate each epoch
            if self.lr_scheduler is not None and not stopped_early:
                self.lr_scheduler.step()

            if stopped_early:
                break

        # After training, save the records
        # stacks Record list to numpy when finished train process
        for k in self.Records.keys():
            self.Records[k] = np.array(self.Records[k])
        for k in self.MinorRecords.keys():
            self.MinorRecords[k] = np.array(self.MinorRecords[k])
        # save the records
        io.savemat(os.path.join(self.save_dir, "Records.mat"), self.Records)
        io.savemat(os.path.join(self.save_dir, "MinorRecords.mat"), self.MinorRecords)

        # log the best and final acc and loss
        final_len = int(max(min(args.max_epoch * 0.5, 5), 1))
        info = (
            "max train acc in epoch {:2d}: {:10.6f}\n".format(
                self.Records["train_acc"].argmax() + 1, self.Records["train_acc"].max()
            )
            + "max val acc in epoch {:2d}: {:10.6f}\n".format(
                self.Records["val_acc"].argmax() + 1, self.Records["val_acc"].max()
            )
            + "final train acc: %.6f\n final val acc: %.6f\n"
            % (self.Records["train_acc"][-final_len:].mean(), self.Records["val_acc"][-final_len:].mean())
        )
        for item in info.split("\n"):
            logging.info(item)
        with open(os.path.join(self.save_dir, "acc output.txt"), "w") as f:
            f.write(info)

        if args.data_name == "THU_Motor_speed" and self.datasets.get("test") is not None:
            best_model_path = os.path.join(
                self.save_dir, f'epoch{self.Records["best_epoch"]}-acc{best_acc*100:.4f}-best_model.pth'
            )
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path)
                self.model.load_state_dict(checkpoint["state_dict"])
                logging.info(f"Loaded best model from {best_model_path}")
            else:
                logging.warning("Best model file not found, using current model.")

            self.model.eval()
            test_acc = 0.0
            with torch.no_grad():
                for inputs, labels in self.dataloaders["test"]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    logits = self.model(inputs)
                    pred = (torch.sigmoid(logits) > 0.5).float()
                    correct = (pred == labels).all(dim=1).float().sum().item()
                    test_acc += correct
            test_acc /= len(self.dataloaders["test"].dataset)
            logging.info(f"Test subset accuracy (exact match) using best model: {test_acc:.4f}")

    def plot_save(self):
        # set color
        current_cmap = sns.color_palette("husl", 10)
        sns.set_theme(style="white")
        sns.set_theme(style="ticks", context="notebook", font="Times New Roman", palette=current_cmap, font_scale=1.5)

        # make dir
        self.save_dir_sub = os.path.join(self.save_dir, "postprosess")
        if not os.path.exists(self.save_dir_sub):
            os.makedirs(self.save_dir_sub)

        # plot MinorRecords
        actual_max_epoch = len(self.Records["train_acc"])
        steps = np.arange(self.MinorRecords["train_loss"].shape[0]) + 1
        steps = steps / len(steps) * actual_max_epoch
        fig = plt.figure(figsize=[10, 8], dpi=100)
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(steps, self.MinorRecords["train_loss"], "b-.d", markersize=3, linewidth=2)
        ax1.set_xlabel("epoch", fontfamily="monospace")
        ax1.set_ylabel("loss", fontfamily="monospace")
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(steps, self.MinorRecords["train_acc"] * 100, "b-.d", markersize=3, linewidth=2)
        ax2.legend(["train acc", "val acc"], loc=5)
        ax2.set_xlabel("epoch", fontfamily="monospace")
        ax2.set_ylabel("acc", fontfamily="monospace")
        ax2.set_ylim([80, 100])
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir_sub, "Minor_loss_acc.jpg"))
