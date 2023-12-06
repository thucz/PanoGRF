from __future__ import absolute_import, division, print_function
import os

import numpy as np
import time
import json
import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
torch.manual_seed(100)
torch.cuda.manual_seed(100)

import datasets
from networks import UniFuse, Equi
from metrics import compute_depth_metrics, Evaluator
from losses import BerhuLoss

class Trainer:
    def __init__(self, settings):
        self.settings = settings

        self.device = torch.device("cuda" if len(self.settings.gpu_devices) else "cpu")
        self.gpu_devices = ','.join([str(id) for id in settings.gpu_devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_devices

        self.log_path = os.path.join(self.settings.log_dir, self.settings.model_name)

        # checking the input height and width are multiples of 32
        assert self.settings.height % 32 == 0, "input height must be a multiple of 32"
        assert self.settings.width % 32 == 0, "input width must be a multiple of 32"

        # data
        datasets_dict = {"3d60": datasets.ThreeD60,
                         "panosuncg": datasets.PanoSunCG,
                         "stanford2d3d": datasets.Stanford2D3D,
                         "matterport3d": datasets.Matterport3D}
        self.dataset = datasets_dict[self.settings.dataset]
        self.settings.cube_w = self.settings.height//2

        fpath = os.path.join(os.path.dirname(__file__), "datasets", "{}_{}.txt")

        train_file_list = fpath.format(self.settings.dataset, "train")
        val_file_list = fpath.format(self.settings.dataset, "val")

        train_dataset = self.dataset(self.settings.data_path, train_file_list, self.settings.height,
                                     self.settings.width,
                                     self.settings.disable_color_augmentation,
                                     self.settings.disable_LR_filp_augmentation,
                                     self.settings.disable_yaw_rotation_augmentation, is_training=True)
        self.train_loader = DataLoader(train_dataset, self.settings.batch_size, True,
                                       num_workers=self.settings.num_workers, pin_memory=True, drop_last=True)
        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // self.settings.batch_size * self.settings.num_epochs

        val_dataset = self.dataset(self.settings.data_path, val_file_list, self.settings.height, self.settings.width,
                                   self.settings.disable_color_augmentation, self.settings.disable_LR_filp_augmentation,
                                   self.settings.disable_yaw_rotation_augmentation, is_training=False)
        self.val_loader = DataLoader(val_dataset, self.settings.batch_size, False,
                                     num_workers=self.settings.num_workers, pin_memory=True, drop_last=True)
        # network
        Net_dict = {"UniFuse": UniFuse,
                    "Equi": Equi}
        Net = Net_dict[self.settings.net]

        self.model = Net(self.settings.num_layers, self.settings.height, self.settings.width,
                         self.settings.imagenet_pretrained, train_dataset.max_depth_meters,
                         fusion_type=self.settings.fusion, se_in_fusion=self.settings.se_in_fusion)

        self.model.to(self.device)
        self.parameters_to_train = list(self.model.parameters())

        self.optimizer = optim.Adam(self.parameters_to_train, self.settings.learning_rate)

        if self.settings.load_weights_dir is not None:
            self.load_model()

        print("Training model named:\n ", self.settings.model_name)
        print("Models and tensorboard events files are saved to:\n", self.settings.log_dir)
        print("Training is using:\n ", self.device)

        self.compute_loss = BerhuLoss()
        self.evaluator = Evaluator()

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.save_settings()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        self.validate()
        for self.epoch in range(self.settings.num_epochs):
            self.train_one_epoch()
            self.validate()
            if (self.epoch + 1) % self.settings.save_frequency == 0:
                self.save_model()

    def train_one_epoch(self):
        """Run a single epoch of training
        """
        self.model.train()

        pbar = tqdm.tqdm(self.train_loader)
        pbar.set_description("Training Epoch_{}".format(self.epoch))

        for batch_idx, inputs in enumerate(pbar):

            outputs, losses = self.process_batch(inputs)

            self.optimizer.zero_grad()
            losses["loss"].backward()
            self.optimizer.step()

            # log less frequently after the first 1000 steps to save time & disk space
            early_phase = batch_idx % self.settings.log_frequency == 0 and self.step < 1000
            late_phase = self.step % 1000 == 0

            if early_phase or late_phase:

                pred_depth = outputs["pred_depth"].detach()
                gt_depth = inputs["gt_depth"]
                mask = inputs["val_mask"]

                depth_errors = compute_depth_metrics(gt_depth, pred_depth, mask)
                for i, key in enumerate(self.evaluator.metrics.keys()):
                    losses[key] = np.array(depth_errors[i].cpu())

                self.log("train", inputs, outputs, losses)

            self.step += 1

    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            if key not in ["rgb", "cube_rgb"]:
                inputs[key] = ipt.to(self.device)

        losses = {}

        equi_inputs = inputs["normalized_rgb"]

        cube_inputs = inputs["normalized_cube_rgb"]

        outputs = self.model(equi_inputs, cube_inputs)

        losses["loss"] = self.compute_loss(inputs["gt_depth"],
                                           outputs["pred_depth"],
                                           inputs["val_mask"])

        return outputs, losses

    def validate(self):
        """Validate the model on the validation set
        """
        self.model.eval()

        self.evaluator.reset_eval_metrics()

        pbar = tqdm.tqdm(self.val_loader)
        pbar.set_description("Validating Epoch_{}".format(self.epoch))

        with torch.no_grad():
            for batch_idx, inputs in enumerate(pbar):
                outputs, losses = self.process_batch(inputs)
                pred_depth = outputs["pred_depth"].detach()
                gt_depth = inputs["gt_depth"]
                mask = inputs["val_mask"]
                self.evaluator.compute_eval_metrics(gt_depth, pred_depth, mask)

        for i, key in enumerate(self.evaluator.metrics.keys()):
            losses[key] = np.array(self.evaluator.metrics[key].avg.cpu())
        self.log("val", inputs, outputs, losses)
        del inputs, outputs, losses

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.settings.batch_size)):  # write a maxmimum of four images
            writer.add_image("rgb/{}".format(j), inputs["rgb"][j].data, self.step)
            writer.add_image("cube_rgb/{}".format(j), inputs["cube_rgb"][j].data, self.step)
            writer.add_image("gt_depth/{}".format(j),
                             inputs["gt_depth"][j].data/inputs["gt_depth"][j].data.max(), self.step)
            writer.add_image("pred_depth/{}".format(j),
                             outputs["pred_depth"][j].data/outputs["pred_depth"][j].data.max(), self.step)

    def save_settings(self):
        """Save settings to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.settings.__dict__.copy()

        with open(os.path.join(models_dir, 'settings.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, "{}.pth".format("model"))
        to_save = self.model.state_dict()
        # save resnet layers - these are needed at prediction time
        to_save['layers'] = self.settings.num_layers
        # save the input sizes
        to_save['height'] = self.settings.height
        to_save['width'] = self.settings.width
        # save the dataset to train on
        to_save['dataset'] = self.settings.dataset
        to_save['net'] = self.settings.net
        to_save['fusion'] = self.settings.fusion
        to_save['se_in_fusion'] = self.settings.se_in_fusion

        torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model from disk
        """
        self.settings.load_weights_dir = os.path.expanduser(self.settings.load_weights_dir)

        assert os.path.isdir(self.settings.load_weights_dir), \
            "Cannot find folder {}".format(self.settings.load_weights_dir)
        print("loading model from folder {}".format(self.settings.load_weights_dir))

        path = os.path.join(self.settings.load_weights_dir, "{}.pth".format("model"))
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.settings.load_weights_dir, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
