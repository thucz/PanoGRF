# Lint as: python3
"""The M3DReader class parses the Matterport3D data for training, validation, and testing.

Example Usage:

my_reader = M3DReader("m3d")
building_data = my_reader.get_training_tfdata(seq_len=2, reference_idx=-1)
"""

import os
import json
import numpy as np
from helpers import my_helpers
import pyproj
from PIL import Image
import torch
import json
from torch.utils.data import Dataset
from PIL import Image
from matplotlib import pyplot as plt


class M3DReader(Dataset):
  """The M3D dataset consists of several panos organized by buildings.

  This class contains functions to split the dataset into training/val/test along the buildings.
  Then we create paths within the buildings to batch consecutive views.

  Args:

  Returns:

  """

  def __init__(self,
               data_path,
               width=512,
               height=512,
               data_type="train",
               scenes=None):
    """Create an M3D reader.

    Args:
      data_path: Path the the unzipped M3D data directory.

    Returns: A new M3D reader.

    """
    super().__init__()

    if data_type not in ["train", "validation", "test"]:
      raise ValueError("Invalid data_type")

    self.data_path = data_path
    self.buildings = os.listdir(self.data_path)
    self.width = width
    self.height = height

    theta, phi = np.meshgrid(
        (np.arange(self.width) + 0.5) * (2 * np.pi / self.width),
        (np.arange(self.height) + 0.5) * (np.pi / self.height))
    uvs, uv_sides = my_helpers.spherical_to_cubemap(
        theta.reshape(-1), phi.reshape(-1))
    self.skybox_uvs = uvs.reshape(self.height, self.width, 2)
    self.skybox_uv_sides = uv_sides.reshape(self.height, self.width)

    if type(scenes) is list:
      self.scenes = scenes
    elif data_type == "train":
      with open(os.path.join(data_path, "training.json"), "r") as f:
        self.scenes = [x for x in json.load(f) if "." not in x]
    elif data_type == "validation":
      with open(os.path.join(data_path, "validation.json"), "r") as f:
        self.scenes = [x for x in json.load(f) if "." not in x]
    elif data_type == "test":
      with open(os.path.join(data_path, "testing.json"), "r") as f:
        self.scenes = [x for x in json.load(f) if "." not in x]

    self.panos = []
    for scene in self.scenes:
      panos = os.listdir(os.path.join(data_path, scene, "equirectangular_depth"))
      panos = [x.split(".")[0] for x in panos]
      for pano in panos:
        self.panos.append((scene, pano))

  def __len__(self):
    """Get length of the data."""
    return len(self.panos)

  def load_pano(self, scene, pano):
    pano_path = os.path.join(self.data_path, scene, "equirectangular",
                             pano + ".jpeg")
    img = Image.open(pano_path)
    return np.array(img, dtype=np.float32) / 255

  def load_depth(self, scene, pano):
    pano_path = os.path.join(self.data_path, scene, "equirectangular_depth",
                             pano + ".png")
    img = Image.open(pano_path)
    img = np.array(img, dtype=np.float32) / 4000
    return img

  def __getitem__(self, idx):
    """Gets a single pano from the dataset.

    Args:
      idx: Index of the item in the dataset.

    Returns:
      Dictionary including pano, depth, rotation, and translation
    """
    if torch.is_tensor(idx):
      idx = idx.tolist()

    m_pano = self.load_pano(self.panos[idx][0], self.panos[idx][1])
    m_depth = self.load_depth(self.panos[idx][0], self.panos[idx][1])
    sample = {"panos": m_pano, "depths": m_depth}
    return sample


if __name__ == "__main__":
  reader = M3DReader(
      "/media/david/David_SSD/matterport3d/stitched", data_type="train")
  print("Len", len(reader))
  plt.imshow(reader[5]["panos"])
  plt.imshow(reader[5]["depths"])
