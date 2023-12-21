"""Train depth and pose on MP3D.
"""
import os
import distro
import numpy as np
# Pytorch Imports
import torch
import argparse
from progress.bar import Bar
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_readers.habitat_data_neuray_ft import HabitatImageGeneratorFT
from lmdb_rw.habitat_data_neuray_ft_lmdb import HabitatImageGeneratorFT_LMDB
from helpers import my_torch_helpers
from helpers.torch_checkpoint_manager import CheckpointManager
from models import loss_lib
from models.test_models import ConvBlock
import numpy as np
import random
from torchvision import transforms
import sys
sys.path.append("./UniFuse-Unidirectional-Fusion/UniFuse")
from datasets.util import Equirec2Cube
import torchvision.utils as vutils 
import cv2
import matplotlib
import yaml

def setup_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True


def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def colorize(value, vmin=None, vmax=None, cmap='jet'):
    value = value.detach().cpu().numpy()[0,:,:]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)

    img = value[:,:,:3]

    return img.transpose((2,0,1))


class App:
  """Main app class"""
  default_cfg={
    "script_mode": "train_depth_pose",
    "model_name": "",
    "checkpoints_dir": "",
    "batch_size": 2,
    "epochs": 999999,#useless
    "learning_rate":0.00006,
    "width": 512,
    "height": 256,
    "opt_beta1": 0.5,
    "opt_beta2": 0.999,
    "train_tensorboard_interval": 0,
    "validation_interval": 200,
    "checkpoint_interval":200,
    "smoothness_loss_lambda": 1,
    "checkpoint_count": 3,
    "point_radius": 0.004,
    "device": "cuda",
    "verbose": True,
    "patch_loss_patch_size": 5,
    "patch_loss_stride": 1,
    "patch_loss_stride_dist": 3,
    "inpaint_use_residual": False,
    "inpaint_wrap_padding": False,
    "loss": "l1",
    "inpaint_use_batchnorm":True,
    "upscale_point_cloud": True,
    "inpaint_one_conv": False,
    "interpolation_mode": "bilinear",
    "add_depth_noise": 0,
    "depth_input_uv": False,
    "normalize_depth": False, 
    "predict_zdepth": False,
    "cost_volume": "",
    "clip_grad_value": 0,
    "model_use_v_input": False,
    "debug_mode": False,
    "dataset": "m3d",
    "carla_min_dist": 2, 
    "carla_max_dist": 100,
    "min_depth": 0.1,
    "max_depth": 10,
    "turbo_cmap_min": 2,
    "m3d_dist": 1,
    "depth_type": "one_over",
    "depth_loss": "l1",
    "load_prepared_test_data": False,
    "test_sample_num": 200,
    "save_datadir": "/home/chenzheng/nas/PanoNVS/somsi_data/test_m3d",
    "total_iter": 100000,
    #unifuse
    "num_layers": 18, # choices=[2, 18, 34, 50, 101]
    "imagenet_pretrained": False,
    # ablation settings
    "net": "UniFuse",#choices=["UniFuse", "Equi", "ERP+TP", "TP", "Cube", "OmniSyn"]
    "fusion":"cee", #choices=["cee", "cat", "biproj"]
    "se_in_fusion": False,
    "num_workers": 2,
    # Multi_view matching hyper_parameters
    "MAGNET_sampling_range": 3, 
    "MAGNET_num_samples": 5,
    "DNET_ckpt": None,
    "contain_dnet": False,
    "use_wrap_padding": False,
    "stereo_out_type": "disparity",
    "dnet_out_type": "depth",
    "out_type": "depth",
    "stereonet_ckpt": None,
    "fuse_type": "simple", #simple, geometry_emb_only, geometry_emb_scaled, geometry_emb_scaled_masked etc
    "patchsize": (128, 128),
    "fov": 80, # unused
    "nrow": 4, # unused
    "median_scaling": False, 
    "mono_uncertainty": False, 
    "mono_lowres_pred": False, 
    "lr_decay": False, 
    "lrate_decay": 250,
    "omnifusion":False,
    "use_lmdb": False,
    "mono_uncert_tune": False,
    "berhuloss": False,
    "freeview": False,
    "offset": [0, 0, 0]
  }
  

  def __init__(self):
    self.model = None
    self.optimizer = None
    self.checkpoint_manager = None
    self.args = None
    self.writer = None
    self.loss_fn = loss_lib.BerhuLoss() 
    # Attributes to hold training data.
    self.train_data = None
    self.train_data_loader = None
    # Attributes to hold validation data.
    self.val_data = None
    self.val_data_indices = None
    self.input_panos_val = None
    self.input_depths_val = None
    self.input_rots_val = None
    self.input_trans_val = None
    self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  def e2c_process(self, panos_small, e2c_instance):
      batch_size, seq_len = panos_small.shape[:2]
      pano_cube_all = []
      for i in range(batch_size):
          pano_cube_seq = []
          for j in range(seq_len):                        
              cube = e2c_instance.run(panos_small[i, j].data.cpu().numpy())
              pano_cube_seq.append(cube)
          pano_cube_all.append(pano_cube_seq)
      pano_cube_all = np.array(pano_cube_all)
      pano_cube_all = torch.from_numpy(pano_cube_all).to(self.args["device"])

      return pano_cube_all
  def normalize_input(self, panos_small, panos_small_cube):      
      batch_size, seq_len = panos_small.shape[:2]
      normalized_equi = []
      normalized_cube = []
      for i in range(batch_size):
          pano_cube_seq = []
          pano_equi_seq = []
          for j in range(seq_len): 
              pano_equi_seq.append(self.normalize(panos_small[i, j]).data.cpu().numpy())
              pano_cube_seq.append(self.normalize(panos_small_cube[i, j]).data.cpu().numpy())
          normalized_equi.append(pano_equi_seq)
          normalized_cube.append(pano_cube_seq)
      normalized_equi = torch.from_numpy(np.array(normalized_equi)).to(self.args["device"])
      normalized_cube = torch.from_numpy(np.array(normalized_cube)).to(self.args["device"])
      return normalized_equi, normalized_cube
  def normalize_erp_input(self, panos_small):      
      batch_size, seq_len = panos_small.shape[:2]
      normalized_equi = []
      for i in range(batch_size):
          pano_equi_seq = []
          for j in range(seq_len): 
              pano_equi_seq.append(self.normalize(panos_small[i, j]).data.cpu().numpy())
          normalized_equi.append(pano_equi_seq)

      normalized_equi = torch.from_numpy(np.array(normalized_equi)).to(self.args["device"])
      return normalized_equi 

  def start(self, flags):
    """Starts the training."""
    try:
      cfg=load_cfg(flags.cfg)
      self.cfg={**self.default_cfg,**cfg}           
      args = self.cfg
      self.args = args
      self.full_width = args["width"]
      self.full_height = args["height"]
      self.e2c = Equirec2Cube(args["mono_height"], args["mono_width"], args["mono_height"] // 2)
      self.load_training_data()
      self.load_validation_data()
      self.setup_model()
      self.setup_checkpoints()
      step = self.checkpoint_manager.step
      
      seed = 2022
      setup_seed(seed)
      if args["script_mode"] == "train_depth_pose":
        self.run_training_loop()
      elif args["script_mode"] == "eval_depth_test":
        self.eval_on_validation_data(step=0)
      else:
        raise ValueError("Unknown script mode: " + str(args["script_mode"]))

    except KeyboardInterrupt:
      print("Terminating script")
      self.writer.close()

  def setup_model(self):
    """Sets up the model."""
    args = self.args
    from select_mononet import select_mono
    model = select_mono(args)
    model.to(args["device"])
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args["learning_rate"], betas=(args["opt_beta1"], args["opt_beta2"])
                                 )
    self.model = model
    self.optimizer = optimizer

  def setup_checkpoints(self):
    """Sets up the checkpoint manager."""
    args = self.args
    model = self.model
    optimizer = self.optimizer

    checkpoint_manager = CheckpointManager(args["checkpoints_dir"],
                                           max_to_keep=args["checkpoint_count"])
    
    latest_checkpoint = checkpoint_manager.load_latest_checkpoint()
    
    if latest_checkpoint is not None:
      model.load_state_dict(latest_checkpoint['model_state_dict'])
      optimizer.load_state_dict(latest_checkpoint['optimizer_state_dict'])
    
    writer = SummaryWriter(log_dir=os.path.join(args["checkpoints_dir"], "logs"))
    self.checkpoint_manager = checkpoint_manager
    self.writer = writer

  def load_training_data(self):
    """Loads training data."""
    args = self.args
    if args["use_lmdb"]:
      seq_len = args["seq_len"]
      reference_idx = args["reference_idx"]
    else:
      seq_len = 1
      reference_idx = 0

    # Prepare dataset loaders for train and validation datasets.
    if args["dataset"] == "m3d":
      if args["use_lmdb"]:
        train_data = HabitatImageGeneratorFT_LMDB(
          args,
          "train",
          full_width=self.full_width,
          full_height=self.full_height,
          m3d_dist=args["m3d_dist"],
          seq_len = seq_len,
          reference_idx = reference_idx,
          aug=args["aug"]
        )
        train_dataloader = DataLoader(
          dataset=train_data,
          num_workers=args["num_workers"],
          batch_size=args["batch_size"],
          shuffle=True,
          drop_last=True,
          pin_memory=True,
          prefetch_factor=1,
        )
      else:
        train_data = HabitatImageGeneratorFT(
          args,
          "train",
          full_width=self.full_width,
          full_height=self.full_height,
          m3d_dist=args["m3d_dist"],
          seq_len = seq_len,
          reference_idx = reference_idx,
          aug=args["aug"]
        )
        train_dataloader = DataLoader(
          dataset=train_data,
          num_workers=args["num_workers"],
          batch_size=args["batch_size"],
          shuffle=False,
          drop_last=True,
          pin_memory=True
        )

    train_data.cache_depth_to_dist(args["height"], args["width"])

    self.train_data = train_data
    self.train_data_loader = train_dataloader

  def load_validation_data(self):
    """Loads validation data."""
    args = self.args
    seq_len = args["val_seq_len"]
    reference_idx = 1

    if args["dataset"] == "m3d":
      mode = "val"
      if "eval_mode" in args and args["eval_mode"] =="test":
        mode= "test"
      if args["use_lmdb"]:
        val_data = HabitatImageGeneratorFT_LMDB(
          args,
          mode,
          full_width=self.full_width,
          full_height=self.full_height,
          m3d_dist=args["m3d_dist"],
          seq_len = seq_len,
          reference_idx = reference_idx)      
      else:
        val_data = HabitatImageGeneratorFT(
          args,
          mode,
          full_width=self.full_width,
          full_height=self.full_height,
          m3d_dist=args["m3d_dist"],
          seq_len = seq_len,
          reference_idx = reference_idx)
    self.val_data = val_data

  def run_depth_pose_carla(self, step, panos, depths, rots, trans):
    """Does a single run and returns results.
    Args:
      step: Current step.
      panos: Input panoramas.
      depths: GT depths.
      rots: GT rotations.
      trans: GT translations.

    Returns:
      Dictionary containing all outputs.

    """

    args = self.args
    height = args["height"]
    width = args["width"]
    model = self.model
    train_data = self.train_data

    batch_size, seq_len = panos.shape[:2]

    panos_small = panos.reshape(
      (batch_size * seq_len, self.full_height, self.full_width, 3))
    
    panos_small = my_torch_helpers.resize_torch_images(
      panos_small, (args["height"], args["width"]), mode=args["interpolation_mode"])

    panos_small = my_torch_helpers.resize_torch_images(
      panos_small, (args["mono_height"], args["mono_width"]), mode=args["interpolation_mode"])
    panos_small = panos_small.reshape(batch_size, seq_len, args["mono_height"], args["mono_width"], 3)

    depths_height, depths_width = depths.shape[2:4]
    
    depths_small = depths.reshape(
      (batch_size * seq_len, depths_height, depths_width, 1))
    depths_small = my_torch_helpers.resize_torch_images(
      depths_small, (args["height"], args["width"]), mode=args["interpolation_mode"])
    depths_small = depths_small.reshape(batch_size, seq_len, height, width, 1)
    if self.args["omnifusion"]:
      pass
    elif self.args["mono_net"]=="UniFuse":
      panos_small_cube = self.e2c_process(panos_small, self.e2c)
      panos_small_cube = panos_small_cube.permute((0, 1, 4, 2, 3))    
    # Change to channels first.
    panos_small = panos_small.permute((0, 1, 4, 2, 3))
    if self.args["mono_net"]=="UniFuse":
      equi_inputs, cube_inputs = self.normalize_input(panos_small, panos_small_cube)
      outputs = model(equi_inputs[:, 0], cube_inputs[:, 0])    
    else:
      raise Exception

    if args["mono_uncertainty"]:
      pred = outputs["pred"]
      pred = torch.nn.functional.interpolate(pred,
                                                (args["height"], args["width"]),
                                                mode=args["interpolation_mode"],
                                                align_corners=False)

      gt_dmap = depths_small.reshape(batch_size, seq_len, height, width).clone()
      gt_dmap[gt_dmap > args["max_depth"]+1] = args["max_depth"] + 1

      depth_loss = loss_lib.loss_uncertainty(
            pred,
            gt_dmap,
            torch.gt(gt_dmap, 0.1),
            sphere=args["sphere"]
      )
      if args['mono_lowres_pred']:
        lowres_pred = torch.nn.functional.interpolate(outputs["lowres_pred"],
                                          (args["height"], args["width"]),
                                          mode=args["interpolation_mode"],
                                          align_corners=False)

        lowres_depth_loss = loss_lib.loss_uncertainty(
              lowres_pred,
              gt_dmap,
              torch.gt(gt_dmap, 0.1),
              sphere=args["sphere"]
        )

        depth_loss = depth_loss+0.5*lowres_depth_loss

      depths_pred = torch.clamp(pred[:, :1, ...], min=0.1) 
      sigma_pred = pred[:, 1:, ...]
      
    else:
      depths_pred = outputs["pred_depth"]
      depths_pred = torch.nn.functional.interpolate(depths_pred,
                                                (args["height"], args["width"]),
                                                mode=args["interpolation_mode"],
                                                align_corners=False)
      if args["mono_uncert_tune"]:
        sigma_pred = outputs["mono_std"]
        sigma_pred = torch.nn.functional.interpolate(sigma_pred,
                                                (args["height"], args["width"]),
                                                mode=args["interpolation_mode"],
                                                align_corners=False)

      gt_dmap = depths_small.reshape(batch_size, seq_len, height, width) 
      gt_dmap[gt_dmap > args["max_depth"]+1] = args["max_depth"] + 1

      if args["mono_uncert_tune"]:
        depth_loss = loss_lib.new_compute_gaussian_loss(depths_pred, gt_dmap, sigma_pred, torch.gt(gt_dmap, 0.1))#0.01 * torch.nn.functional.gaussian_nll_loss()
      else:
        if args["berhuloss"]:
          gt_dmap_mask = torch.gt(gt_dmap, 0.1)
          depth_loss = self.loss_fn(gt_dmap, depths_pred, gt_dmap_mask)#outputs, gt_dmap, gt_dmap_mask)
        else:
          depth_loss = loss_lib.compute_l1_sphere_loss(
              depths_pred.permute((0, 2, 3, 1)),
              gt_dmap.permute((0, 2, 3, 1)),
              torch.gt(gt_dmap.permute((0, 2, 3, 1)), 0.1)
          )
    depths_pred = torch.clamp(depths_pred, min=0.1)
    final_loss = depth_loss
    ret_data = {
      "final_loss": final_loss,
      "depths_pred": depths_pred,
      "panos_small": panos_small,
      "depths_small": depths_small,
      "outputs": outputs,
    }
    if args["mono_uncertainty"] or args["mono_uncert_tune"]:
      ret_data["sigma_pred"] = sigma_pred
    return ret_data 
  def do_validation_run(self, step):
    """Does a validation run.
    Args:
      step: Current step.
    Returns:
      None.
    """
    args = self.args
    model = self.model
    writer = self.writer
    if step == 1 or \
        args["validation_interval"] == 0 or \
        step % args["validation_interval"] == 0:
      # Calculate validation final_loss.
      with torch.no_grad():
        panos = self.input_panos_val
        depths = self.input_depths_val
        rots = self.input_rots_val
        trans = self.input_trans_val

        panos = panos[:, 1:, ...]
        depths = depths[:, 1:, ...]
        trans = trans[:, 1:, ...]
        rots = rots[:, 1:, ...]

        run_outputs = self.run_depth_pose_carla(step, panos, depths, rots,
                                                trans)
        final_loss = run_outputs["final_loss"]
        depths_pred = run_outputs["depths_pred"]
        depths_small = run_outputs["depths_small"]
        panos_small = run_outputs["panos_small"]

        writer.add_scalar("val_loss", final_loss.item(), step)

  def log_training_to_tensorboard(self, step, run_outputs, mode='train'):
    """Logs training to tensorboard.

    Args:
      step: Current step.
      run_outputs: Outputs of the training step.

    Returns:
      None.

    """
    args = self.args
    model = self.model
    writer = self.writer

    depths_pred = run_outputs["depths_pred"]
    depths_small = run_outputs["depths_small"]
    outputs = run_outputs["outputs"]
    panos_small = run_outputs["panos_small"]

    final_loss = run_outputs["final_loss"]
    loss_np = final_loss.detach().cpu().numpy()
    average_depth_np = torch.mean(depths_pred).detach().cpu().numpy()

    writer.add_scalar("train_loss", loss_np, step)
    writer.add_scalar("train_depth", average_depth_np, step)

    if step == 1 or \
        args["train_tensorboard_interval"] == 0 or \
        step % args["train_tensorboard_interval"] == 0:
      with torch.no_grad():
        tmp_pano = panos_small[:, 0].data#.permute(0, 3, 1, 2).data
        
        tmp_depth = depths_small[:, 0].permute(0, 3, 1, 2).data
        tmp_depth_pred = depths_pred.data #only one data.
        writer.add_image(mode+'_RGB', vutils.make_grid(tmp_pano, nrow=4, normalize=True), step)   
        writer.add_image(mode+'_depth gt', colorize(vutils.make_grid(tmp_depth, nrow=4, normalize=False)), step)
        writer.add_image(mode+'_depth pred', colorize(vutils.make_grid(tmp_depth_pred, nrow=4, normalize=False)), step)
        
        

  def save_checkpoint(self, step):
    """Saves a checkpoint.

    Args:
      step: Current step.

    Returns:
      None.

    """
    args = self.args

    if args["checkpoint_interval"] == 0 or step % args["checkpoint_interval"] == 0:
      # Save a checkpoint
      self.checkpoint_manager.save_checkpoint({
        "model_state_dict": self.model.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict()
      })
      self.writer.flush()

  def run_training_loop(self):
    args = self.args
    train_dataloader = self.train_data_loader
    optimizer = self.optimizer
    checkpoint_manager = self.checkpoint_manager
    model = self.model

    for epoch in range(args["epochs"]):
      for i, data in enumerate(train_dataloader):        
        print('i:', i)
        optimizer.zero_grad()
        step = checkpoint_manager.increment_step()
          
        if step % args["checkpoint_interval"]==0 and step>0:
          self.save_checkpoint(step)          
        
        if step >= args["total_iter"]:

          print("evaluation...")
          self.save_checkpoint(step)
          self.eval_on_validation_data(step)
          exit()
        
        if step % args["validation_interval"]==0 and step>0:
          self.eval_on_validation_data(step)
        if args["debug_mode"]:
          assert distro.linux_distribution()[0] == "Ubuntu", "Debug mode is on"
          panos = self.input_panos_val
          depths = self.input_depths_val
          rots = self.input_rots_val
          trans = self.input_trans_val
        else:
          assert distro is not None
          if args["use_lmdb"]:
            panos = data["rgb_panos"][:, 1:, ...].to(args["device"])
            depths = data["depth_panos"][:, 1:, ...].to(args["device"])
            rots = data["rots"][:, 1:, ...].to(args["device"])
            trans = data["trans"][:, 1:, ...].to(args["device"])

          else:
            panos = data["rgb_panos"].to(args["device"])
            depths = data["depth_panos"].to(args["device"])
            rots = data["rots"].to(args["device"])
            trans = data["trans"].to(args["device"])
        assert panos.shape[1] <= 1
        run_outputs = self.run_depth_pose_carla(step, panos, depths,
                                                rots,
                                                trans)
        self.log_training_to_tensorboard(step, run_outputs, mode='train')

        final_loss = run_outputs["final_loss"]
        depths_pred = run_outputs["depths_pred"]
        if self.args["lr_decay"]:
            if self.args["grad_scale"]:
              self.scaler.scale(final_loss).backward()
              self.scaler.unscale_(self.optimizer)
              torch.nn.utils.clip_grad_norm_(model.parameters(), args["clip_grad_value"])
              self.scaler.step(self.optimizer)
              self.scaler.update()           
              self.optimizer.step()
            else:
              final_loss.backward()
              torch.nn.utils.clip_grad_value_(model.parameters(), args["clip_grad_value"])
              self.optimizer.step()
            self.save_checkpoint(step)

            loss_np = final_loss.detach().cpu().numpy()
            average_depth_np = torch.mean(depths_pred).detach().cpu().numpy()
            print("Step: %d [%d:%d] Loss: %f, average depth %f" %
                  (step, epoch, i, loss_np, average_depth_np))


            # self.scheduler.step()
            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            decay_rate = 0.1
            #250*1000=250000
            decay_steps = args["lrate_decay"] * 1000 # iteration: 20w->10w, 1000->500
            new_lrate = args["learning_rate"] * (decay_rate ** (step / decay_steps))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lrate

        else:
          final_loss.backward()
          if args["clip_grad_value"] > 1e-10:
            torch.nn.utils.clip_grad_value_(model.parameters(),
                                            args["clip_grad_value"])

          optimizer.step()
          self.save_checkpoint(step)
          loss_np = final_loss.detach().cpu().numpy()
          average_depth_np = torch.mean(depths_pred).detach().cpu().numpy()
          if step % 100 == 0:
            print("Step: %d [%d:%d] Loss: %f, average depth %f" %
                (step, epoch, i, loss_np, average_depth_np))
    if args["use_lmdb"]:
      self.train_data.env.close()
  def eval_on_training_data(self):
    """Performs evaluation on the whole evaluation dataset.

    Returns:
      None
    """
    args = self.args
    train_dataloader = DataLoader(self.train_data,
                                  batch_size=args["batch_size"],
                                  shuffle=False,
                                  num_workers=4,
                                  prefetch_factor=1)

    self.model.eval()

    num_iterations = 10
    bar = Bar('Eval on training data',
              max=num_iterations)

    train_iterator = iter(train_dataloader)

    min_pred_depth = 9999.9
    max_pred_depth = 0.0
    min_gt_depth = 999.9
    max_gt_depth = 0.0
    l1_errors = []
    l2_errors = []
    wl1_errors = []
    wl2_errors = []
    with torch.no_grad():
      step = 100000
      weight = (torch.arange(0, args["height"], device=args["device"],
                             dtype=torch.float32) + 0.5) * np.pi / args["height"]
      weight = torch.sin(weight).view(1, args["height"], 1, 1)
      for i in range(num_iterations):
        data = next(train_iterator)
        panos = data["rgb_panos"].to(args["device"])
        depths = data["depth_panos"].to(args["device"])
        rots = data["rots"].to(args["device"])
        trans = data["trans"].to(args["device"])

        run_outputs = self.run_depth_pose_carla(step, panos, depths,
                                                rots,
                                                trans)

        m_weight = weight.expand(
          panos.shape[0], args["height"], args["width"], 1)

        depths_small = run_outputs["depths_small"][:, 1]
        depths_pred = run_outputs["depths_pred"][:, 0]
        depths_pred = torch.clamp_min(depths_pred, 0.0)

        wl1_error = torch.abs(depths_small - depths_pred) * m_weight
        wl1_error = torch.sum(wl1_error, dim=(1, 2, 3)) / torch.sum(m_weight,
                                                                    dim=(
                                                                      1, 2, 3))
        wl1_errors.append(wl1_error.cpu().numpy())

        wl2_error = torch.pow(depths_small - depths_pred, 2.0) * m_weight
        wl2_error = torch.sum(wl2_error, dim=(1, 2, 3)) / torch.sum(m_weight,
                                                                    dim=(
                                                                      1, 2, 3))
        wl2_errors.append(wl2_error.cpu().numpy())

        l1_error = torch.mean(torch.abs(depths_small - depths_pred),
                              dim=(1, 2, 3))
        l1_errors.append(l1_error.cpu().numpy())

        l2_error = torch.mean(torch.pow(depths_small - depths_pred, 2.0),
                              dim=(1, 2, 3))
        l2_errors.append(l2_error.cpu().numpy())

        min_pred_depth = min(min_pred_depth, torch.min(depths_pred).item())
        max_pred_depth = max(max_pred_depth, torch.max(depths_pred).item())
        min_gt_depth = min(min_gt_depth, torch.min(depths_small).item())
        max_gt_depth = max(max_gt_depth, torch.max(depths_small).item())
        bar.next()
    total_l1_errors = np.mean(np.stack(l1_errors))
    total_l2_errors = np.mean(np.stack(l2_errors))
    total_wl1_errors = np.mean(np.stack(wl1_errors))
    total_wl2_errors = np.mean(np.stack(wl2_errors))
    bar.finish()
    print("Evaluation on training data:")
    print("Total l1 error:", total_l1_errors, "Weighted:", total_wl1_errors)
    print("Total l2 error:", total_l2_errors, "Weighted:", total_wl2_errors)
    print("True depth range", min_gt_depth, max_gt_depth)
    print("Pred depth range", min_pred_depth, max_pred_depth)

  def eval_on_validation_data(self, step=0):
    """Performs evaluation on the whole evaluation dataset.

    Returns:
      None
    """
    args = self.args

    self.model.eval()
    self.load_validation_data()

    val_dataloader = DataLoader(self.val_data,
                                batch_size=1,
                                shuffle=False,
                                num_workers=0 if args["dataset"] == 'm3d' else 4,
                                pin_memory=True)
    max_examples = len(self.val_data)
    if args["dataset"] == 'm3d':
      max_examples = min(len(self.val_data), 5000)

    if args["script_mode"] == 'eval_depth_test':
      results_file = os.path.join(
        args["checkpoints_dir"],
        "test_eval_results_%02f_%02f.txt" % (
          args["carla_min_dist"], args["carla_max_dist"]))
      if args["dataset"] == 'm3d':
        results_file = os.path.join(
          args["checkpoints_dir"],
          "test_eval_results_%02f.txt" % (
            args["m3d_dist"]))
      results_file = open(results_file, "w")
    else:
      results_file = os.path.join(
        args["checkpoints_dir"],
        "val_eval_results_%02f_%02f.txt" % (
          args["carla_min_dist"], args["carla_max_dist"]))
      if args["dataset"] == 'm3d':
        results_file = os.path.join(
          args["checkpoints_dir"],
          "val_eval_results_%02f_%06d.txt" % (
            args["m3d_dist"], step))
      results_file = open(results_file, "w")

    min_pred_depth = 9999.9
    max_pred_depth = 0.0
    min_gt_depth = 999.9
    max_gt_depth = 0.0
    all_errors = {}

    with torch.no_grad():
      weight = (torch.arange(0, args["height"], device=args["device"],
                             dtype=torch.float32) + 0.5) * np.pi / args["height"]
      weight = torch.sin(weight).view(1, args["height"], 1, 1)
      def load_data(args, idx):
        data = np.load(args["save_datadir"]+'/data_'+str(idx)+'.npz')  
        return data

      if step != 0:
        test_imgs_dir=os.path.join(args["checkpoints_dir"], "test_images_"+str(step)+"_"+str(args["m3d_dist"]))
      else:
        test_imgs_dir=os.path.join(args["checkpoints_dir"], "test_images"+"_"+str(args["m3d_dist"]))
      os.makedirs(test_imgs_dir, exist_ok=True)


      if args["load_prepared_test_data"]:
        for i in range(args["test_sample_num"]):
          print("evaluate i:", i)
          data = load_data(args, i)
          panos = torch.from_numpy(data['panos'])[:, 1:, ...].to(args["device"])
          rots = torch.from_numpy(data['rots'])[:, 1:, ...].to(args["device"])
          trans = torch.from_numpy(data['trans'])[:, 1:, ...].to(args["device"])
          depths = torch.from_numpy(data['depths'])[:, 1:, ...].to(args["device"])
          run_outputs = self.run_depth_pose_carla(step, panos, depths,
                                                  rots,
                                                  trans)

          m_weight = weight.expand(
            panos.shape[0], args["height"], args["width"], 1)
          depths_small = run_outputs["depths_small"][:, 0]

          depths_pred = run_outputs["depths_pred"][:, 0].unsqueeze(3)
          depths_pred = torch.clamp_min(depths_pred, 0.0)

          def median_scaling(pred, gt):
            mask = gt > args["min_depth"] 
            pred = pred[mask]
            gt = gt[mask]
            ratio = np.median(gt) / \
                        np.median(pred)

            return ratio

          erp_errors = self.compute_erp_depth_results(
            gt_depth=depths_small,
            pred_depth=depths_pred,
            m_weight=m_weight
          )
          cube_errors = self.compute_zdepth_results(
            gt_depth=depths[:, 0, :, :, None],
            pred_depth=depths_pred
          )

          for k, v in erp_errors.items():
            if k not in all_errors:
              all_errors[k] = []
            all_errors[k].append(v.detach().cpu().numpy())

          for k, v in cube_errors.items():
            if k not in all_errors:
              all_errors[k] = []
            all_errors[k].append(v.detach().cpu().numpy())

      else:
        for i, data in enumerate(val_dataloader):
          print("i:", i)
          if i >= args["test_sample_num"]:
            break
          if args["val_seq_len"]>=3:
            panos = data["rgb_panos"][:, 2:, ...].to(args["device"])
            depths = data["depth_panos"][:, 2:, ...].to(args["device"])
            rots = data["rots"][:, 2:, ...].to(args["device"])
            trans = data["trans"][:, 2:, ...].to(args["device"])

          else:
            panos = data["rgb_panos"][:, 1:, ...].to(args["device"])
            depths = data["depth_panos"][:, 1:, ...].to(args["device"])
            rots = data["rots"][:, 1:, ...].to(args["device"])
            trans = data["trans"][:, 1:, ...].to(args["device"])

          run_outputs = self.run_depth_pose_carla(step, panos, depths,
                                                  rots,
                                                  trans)

          m_weight = weight.expand(
            panos.shape[0], args["height"], args["width"], 1)

          depths_small = run_outputs["depths_small"][:, 0] 
          depths_pred = run_outputs["depths_pred"][:, 0]
          if args["mono_uncertainty"] or args["mono_uncert_tune"]:
            sigma_pred = run_outputs["sigma_pred"][:, 0]

          depths_pred = torch.clamp_min(depths_pred, 0.0)
        
          def median_scaling(pred, gt):
            mask = gt > args["min_depth"] 
            pred = pred[mask]
            gt = gt[mask]
            ratio = np.median(gt) / \
                        np.median(pred)
            return ratio
          if self.args["median_scaling"]:
            ratio = median_scaling(depths_pred[..., None].data.cpu().numpy(), depths_small.data.cpu().numpy())
            depths_pred = depths_pred * ratio
          rgb = np.uint8(panos[:, 0][0].data.cpu().numpy()*255) 
          rgb = cv2.resize(rgb, (args["width"], args["height"]), cv2.INTER_LINEAR)
          d_pred = depths_pred[0].data.cpu().numpy()
          d_gt = depths_small[0].data.cpu().numpy()
          if args["mono_uncertainty"] or args["mono_uncert_tune"]:
            sig_pred = sigma_pred[0].data.cpu().numpy()
          def normalize_depth(depth_pred, depth_gt):
              d_min = min(depth_pred.min(), depth_gt.min())
              d_max = max(depth_pred.max(), depth_gt.max())
              depth_pred_norm = np.uint8((depth_pred-d_min)/(d_max-d_min)*255)
              depth_gt_norm = np.uint8((depth_gt-d_min)/(d_max-d_min)*255)

              pred_rgb = cv2.applyColorMap(depth_pred_norm, cv2.COLORMAP_JET)
              gt_rgb = cv2.applyColorMap(depth_gt_norm, cv2.COLORMAP_JET)
              return pred_rgb, gt_rgb

          np.savez(test_imgs_dir+"/"+str(i)+"_depths.npz", pred=d_pred, gt=d_gt)
          d_pred, d_gt = normalize_depth(d_pred, d_gt)
          gap = np.zeros((d_gt.shape[0], 10, 3))
          d_vis = np.concatenate([rgb, gap, d_gt, gap, d_pred], axis=1)
          cv2.imwrite(test_imgs_dir+"/"+str(i)+"_depth_vis.jpg", d_vis)
          def normalize_sigma(sigma_pred):
            s_min = sigma_pred.min()
            s_max = sigma_pred.max()
            s_pred_norm = np.uint8((sigma_pred-s_min)/(s_max-s_min)*255)
            pred_rgb = cv2.applyColorMap(s_pred_norm, cv2.COLORMAP_PINK)
            return pred_rgb
          if args["mono_uncertainty"] or args["mono_uncert_tune"]:
            sig_vis = normalize_sigma(sig_pred)
            cv2.imwrite(test_imgs_dir+"/"+str(i)+"_sigma_vis.jpg", sig_vis)
          
          erp_errors = self.compute_erp_depth_results(
            gt_depth=depths_small,
            pred_depth=depths_pred[..., None],
            m_weight=m_weight
          )

          cube_errors = self.compute_zdepth_results(
            gt_depth=depths[:, 0, :, :, None], 
            pred_depth=depths_pred[... ,None]
          )

          for k, v in erp_errors.items():
            if k not in all_errors:
              all_errors[k] = []
            all_errors[k].append(v.detach().cpu().numpy())

          for k, v in cube_errors.items():
            if k not in all_errors:
              all_errors[k] = []
            all_errors[k].append(v.detach().cpu().numpy())

    all_errors_concatenated = {}
    for k, v in all_errors.items():
      all_errors_concatenated[k] = np.mean(np.concatenate(v))

    for k, v in all_errors_concatenated.items():
      results_file.write("%s: %0.5f\n" % (k, v))
    results_file.close()
    if args["use_lmdb"]:
      self.val_data.env.close()
    print("Evaluation done")

  def compute_erp_depth_results(self, gt_depth, pred_depth, m_weight):
    """Computes and returns results.

    Args:
      gt_depth: ERP GT depth.
      pred_depth: ERP predicted euclidean depth.

    Returns:
      Dictionary of torch tensors.

    """
    args = self.args
    valid_regions = torch.logical_and(torch.gt(gt_depth, 0.1),
                                      torch.lt(gt_depth, args["max_depth"]))
    valid_regions_sum = torch.sum(valid_regions, dim=(1, 2, 3))
    m_weight = m_weight * valid_regions
    m_weight_sum = torch.sum(m_weight, dim=(1, 2, 3))
    one_over_gt_depth = my_torch_helpers.safe_divide(1.0, gt_depth)
    one_over_pred_depth = my_torch_helpers.safe_divide(1.0, pred_depth)

    print("Min depth, max depth", torch.min(gt_depth), torch.max(gt_depth))
    print("Min pred depth, max pred depth", torch.min(pred_depth),
          torch.max(pred_depth))

    imae_error = torch.abs(
      one_over_gt_depth - one_over_pred_depth) * valid_regions
    if torch.any(imae_error > 100.0):
      print("max imae", torch.max(imae_error))
      big_error = (imae_error > 100.0).float()
      for i in range(pred_depth.shape[0]):
        my_torch_helpers.save_torch_image(
          os.path.join(args["checkpoints_dir"], "pred_depth_%d.png" % i),
          my_torch_helpers.depth_to_turbo_colormap(
            pred_depth[i:(i + 1)],
            min_depth=args["turbo_cmap_min"]
          )
        )
        my_torch_helpers.save_torch_image(
          os.path.join(args["checkpoints_dir"], "gt_depth_%d.png" % i),
          my_torch_helpers.depth_to_turbo_colormap(
            gt_depth[i:(i + 1)],
            min_depth=args["turbo_cmap_min"]
          )
        )
        my_torch_helpers.save_torch_image(
          os.path.join(args["checkpoints_dir"], "error_depth_%d.png" % i),
          big_error[i:(i + 1)].expand((-1, -1, -1, 3))
        )
      raise ValueError("Error")
    imae_error = torch.abs(
      one_over_gt_depth - one_over_pred_depth) * valid_regions
    imae_error = torch.sum(imae_error, dim=(1, 2, 3)) / valid_regions_sum

    irmse_error = torch.pow(
      one_over_gt_depth - one_over_pred_depth, 2.0) * valid_regions
    irmse_error = torch.sum(irmse_error, dim=(1, 2, 3)) / valid_regions_sum
    irmse_error = torch.sqrt(irmse_error)

    l1_error = torch.abs(gt_depth - pred_depth) * valid_regions
    l1_error = torch.sum(l1_error, dim=(1, 2, 3)) / valid_regions_sum

    l2_error = torch.pow(gt_depth - pred_depth, 2.0)
    l2_error = torch.sum(l2_error, dim=(1, 2, 3)) / valid_regions_sum
    rmse_error = torch.sqrt(l2_error)

    wl1_error = torch.abs(gt_depth - pred_depth) * m_weight
    wl1_error = torch.sum(wl1_error, dim=(1, 2, 3)) / m_weight_sum

    wl2_error = torch.pow(gt_depth - pred_depth, 2.0) * m_weight
    wl2_error = torch.sum(wl2_error, dim=(1, 2, 3)) / m_weight_sum

    wrmse_error = torch.sqrt(wl2_error)

    relative_error = (torch.abs(
      gt_depth - pred_depth) / gt_depth) * valid_regions
    relative_105 = torch.sum(
      torch.logical_and(valid_regions,
                        torch.lt(relative_error, 1.05 - 1)).float(),
      dim=(1, 2, 3)) / valid_regions_sum
    relative_110 = torch.sum(
      torch.logical_and(valid_regions,
                        torch.lt(relative_error, 1.10 - 1)).float(),
      dim=(1, 2, 3)) / valid_regions_sum
    relative_125 = torch.sum(
      torch.logical_and(valid_regions,
                        torch.lt(relative_error, 1.25 - 1)).float(),
      dim=(1, 2, 3)) / valid_regions_sum
    relative_125_2 = torch.sum(
      torch.logical_and(valid_regions,
                        torch.lt(relative_error, 1.25 ** 2 - 1)).float(),
      dim=(1, 2, 3)) / valid_regions_sum
    relative_125_3 = torch.sum(
      torch.logical_and(valid_regions,
                        torch.lt(relative_error, 1.25 ** 3 - 1)).float(),
      dim=(1, 2, 3)) / valid_regions_sum

    return {
      'l1_error': l1_error,
      'l2_error': l2_error,
      'rmse_error': rmse_error,
      'wl1_error': wl1_error,
      'wl2_error': wl2_error,
      'wrmse_error': wrmse_error,
      'relative_105': relative_105,
      'relative_110': relative_110,
      'relative_125': relative_125,
      'relative_125_2': relative_125_2,
      'relative_125_3': relative_125_3,
      'imae_error': imae_error,
      'irmse_error': irmse_error
    }

  def compute_zdepth_results(self, gt_depth, pred_depth,
                             cubemap_sides=(2, 3, 4, 5)):
    """Computes z-depth results on the 4 cubemap sides.

    Args:
      gt_depth: ERP euclidean depth.
      pred_depth: Predicted depth.
      cubemap_sides: Which sides of the cubemap.

    Returns:
      Dictionary of torch tensors.

    """
    args = self.args
    pred_depth_cube = []
    gt_depth_cube = []
    pred_zdepth = self.train_data.distance_to_zdepth_torch(pred_depth)
    gt_zdepth = self.train_data.distance_to_zdepth_torch(gt_depth)
    for side in cubemap_sides:
      pred_depth_cube.append(
        my_torch_helpers.equirectangular_to_cubemap(
          pred_zdepth, side=side))
      gt_depth_cube.append(
        my_torch_helpers.equirectangular_to_cubemap(
          gt_zdepth, side=side))
    pred_zdepth_cube = torch.stack(pred_depth_cube, dim=1)
    gt_zdepth_cube = torch.stack(gt_depth_cube, dim=1)
    valid_regions = torch.logical_and(torch.gt(gt_zdepth_cube, 0.1),
                                      torch.lt(gt_zdepth_cube, args["max_depth"]))
    valid_regions_sum = torch.sum(valid_regions, dim=(1, 2, 3, 4))
    one_over_gt = my_torch_helpers.safe_divide(1.0, gt_zdepth_cube)
    one_over_pred = my_torch_helpers.safe_divide(1.0, pred_zdepth_cube)

    imae_error = torch.abs(one_over_gt - one_over_pred) * valid_regions
    imae_error = torch.sum(imae_error, dim=(1, 2, 3, 4)) / valid_regions_sum

    irmse_error = torch.pow(one_over_gt - one_over_pred, 2.0) * valid_regions
    irmse_error = torch.sum(irmse_error, dim=(1, 2, 3, 4)) / valid_regions_sum
    irmse_error = torch.sqrt(irmse_error)

    l1_error = torch.abs(gt_zdepth_cube - pred_zdepth_cube) * valid_regions
    l1_error = torch.sum(l1_error, dim=(1, 2, 3, 4)) / valid_regions_sum

    l2_error = torch.pow(gt_zdepth_cube - pred_zdepth_cube, 2.0)
    l2_error = torch.sum(l2_error, dim=(1, 2, 3, 4)) / valid_regions_sum
    rmse_error = torch.sqrt(l2_error)

    relative_error = (torch.abs(
      gt_zdepth_cube - pred_zdepth_cube) / gt_zdepth_cube) * valid_regions
    relative_105 = torch.sum(
      torch.logical_and(valid_regions,
                        torch.lt(relative_error, 1.05 - 1)).float(),
      dim=(1, 2, 3, 4)) / valid_regions_sum
    relative_110 = torch.sum(
      torch.logical_and(valid_regions,
                        torch.lt(relative_error, 1.10 - 1)).float(),
      dim=(1, 2, 3, 4)) / valid_regions_sum
    relative_125 = torch.sum(
      torch.logical_and(valid_regions,
                        torch.lt(relative_error, 1.25 - 1)).float(),
      dim=(1, 2, 3, 4)) / valid_regions_sum
    relative_125_2 = torch.sum(
      torch.logical_and(valid_regions,
                        torch.lt(relative_error, 1.25 ** 2 - 1)).float(),
      dim=(1, 2, 3, 4)) / valid_regions_sum
    relative_125_3 = torch.sum(
      torch.logical_and(valid_regions,
                        torch.lt(relative_error, 1.25 ** 3 - 1)).float(),
      dim=(1, 2, 3, 4)) / valid_regions_sum

    return {
      'cube_l1_error': l1_error,
      'cube_l2_error': l2_error,
      'cube_rmse_error': rmse_error,
      'cube_relative_105': relative_105,
      'cube_relative_110': relative_110,
      'cube_relative_125': relative_125,
      'cube_relative_125_2': relative_125_2,
      'cube_relative_125_3': relative_125_3,
      'cube_imae_error': imae_error,
      'cube_irmse_error': irmse_error
    }


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--cfg', type=str, default='configs/train/gen/neuray_gen_depth_train.yaml')
  flags = parser.parse_args()

  app = App()
  app.start(flags)