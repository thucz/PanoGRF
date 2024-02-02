# Lint as: python3
"""This helper class parses the Google Street View data for training.
"""

import os
import json
import numpy as np
from helpers import my_helpers
import pyproj
from PIL import Image
import torch
from torch.utils.data import Dataset


class GSVReader(Dataset):
  """Create a GSVReader by passing in the OS path for the scraped GSV folder."""

  def __init__(self,
               gsv_path,
               width=512,
               height=256,
               data_type="train",
               seq_len=2,
               reference_idx=-1):
    """Create a GSVReader object.

    Args:
      gsv_path: Path to gsv folder.
      width: Width of panos to output.
      height: Height of panos to output.
      data_type: Either train, validation, or test.
      seq_len: Length of each sequence of panos. Should be less than the minimum sequence.
      reference_idx: Index of the reference pano in each sequence. All panos are oriented wrt the reference.
    """
    super().__init__()
    self.gsv_path = gsv_path

    if reference_idx < 0:
      reference_idx = seq_len - 1

    self.width = width
    self.height = height
    self.data_type = data_type
    self.seq_len = seq_len
    self.reference_idx = reference_idx

    if data_type not in ["train", "validation", "test"]:
      raise ValueError("Invalid GSVReader data type: " + str(data_type))
    if reference_idx >= seq_len:
      raise ValueError("Refernece idx %d greater than sequence length %d" %
                       (reference_idx, seq_len))

    # Read the data and process it.
    if data_type == "train":
      with open(os.path.join(gsv_path, "training_paths.json"), "r") as f:
        self.my_paths = json.load(f)
    elif data_type == "validation":
      with open(os.path.join(gsv_path, "validation_paths.json"), "r") as f:
        self.my_paths = json.load(f)
    elif data_type == "test":
      with open(os.path.join(gsv_path, "testing_paths.json"), "r") as f:
        self.my_paths = json.load(f)
    # Load pano metadata.
    with open(os.path.join(gsv_path, "metadata.json"), "r") as f:
      self.pano_metadata = {}
      for k, v in json.load(f):
        self.pano_metadata[k] = v

    self.pano_path = os.path.join(gsv_path, "panos")
    self.geod = pyproj.Geod(ellps="WGS84")

    self.restrided_paths = self.generate_subsequences(self.my_paths, seq_len)
    self.rots, self.trans = self.calculate_rot_trans(
        self.restrided_paths, reference_idx=reference_idx)

  def generate_subsequences(self, sequences, seq_len=2):
    """Takes an array of sequences of arbitrary length and generates
    an array of sequences with fixed length.

    Args:
      sequences: array of sequences with arbitrary length
      seq_len: length of target sequences (Default value = 2)

    Returns:
      array of sequence with length len

    """
    new_paths = []
    for path in sequences:
      path = np.array(path)
      if len(path) >= seq_len:
        shape = (len(path) - seq_len + 1, seq_len)
        strides = (path.itemsize, path.itemsize)
        strided = np.lib.stride_tricks.as_strided(
            path, shape=shape, strides=strides)
        new_paths.append(strided)
    return np.concatenate(new_paths, axis=0)

  def load_and_resize_panos(self, pano_ids):
    """
    Loads and resizes panoramas as float32 tensors.

    Args:
      pano_id: Array of panoids.

    Returns: Resized panos stacked along 0 axis.

    """
    panos = []
    for id in pano_ids:
      full_pano_path = os.path.join(self.pano_path, id + ".png")
      img = Image.open(full_pano_path).resize((self.width, self.height))
      panos.append(np.array(img, dtype=np.float32) / 255)
    return np.stack(panos, axis=0)

  def calculate_rot_trans(self, paths, reference_idx=-1):
    """Calculates rotation and translation of each pano in the path relative to a single pano

    Args:
      paths: Array of pano ids
      reference_idx: Index of the reference pano within each path,
    i.e. the pano with identity rotation, zero translation.
    -1 for the last pano (Default value = -1)

    Returns: rotations and translations

    """
    if reference_idx < 0:
      reference_idx = paths.shape[1] - 1
    rotations = np.zeros((paths.shape[0], paths.shape[1], 3, 3),
                         dtype=np.float32)
    translations = np.zeros((paths.shape[0], paths.shape[1], 3),
                            dtype=np.float32)
    for i, my_path in enumerate(paths):
      reference_metadata = self.pano_metadata[my_path[reference_idx]]
      reference_lat = reference_metadata["location"]["lat"]
      reference_lng = reference_metadata["location"]["lng"]
      reference_heading = reference_metadata["result"]["tiles"]["centerHeading"]
      for j, pano_id in enumerate(my_path):
        if j is reference_idx:
          rot = np.eye(3, dtype=np.float32)
          t = np.zeros(3, dtype=np.float32)
        else:
          my_metadata = self.pano_metadata[pano_id]
          my_lat = my_metadata["location"]["lat"]
          my_lng = my_metadata["location"]["lng"]
          my_heading = my_metadata["result"]["tiles"]["centerHeading"]
          heading_diff = my_helpers.angle_diff_deg(my_heading,
                                                   reference_heading)
          # Applying rot and t to the pano should yield the reference image
          rot = my_helpers.rotate_around_axis(
              np.array([0, 1, 0]), np.deg2rad(heading_diff))
          # forward azimuth [-180, 180], backward azimuth, distance (meters)
          azimuth, bazimuth, distance = self.geod.inv(my_lng, my_lat,
                                                      reference_lng,
                                                      reference_lat)
          t = np.array([
              -np.sin(
                  np.deg2rad(
                      my_helpers.angle_diff_deg(bazimuth, reference_heading))),
              0,
              np.cos(
                  np.deg2rad(
                      my_helpers.angle_diff_deg(bazimuth, reference_heading)))
          ],
                       dtype=np.float32) * distance
        rotations[i, j, :, :] = rot
        translations[i, j, :] = t
    return rotations, translations

  def __len__(self):
    """Get length of the data."""
    return self.restrided_paths.shape[0]

  def __getitem__(self, idx):
    """Gets a single item from the dataset.

    Args:
      idx: Index of the item in the dataset.

    Returns:
      Dictionary including panos, rotation, and translation.
    """
    if torch.is_tensor(idx):
      idx = idx.tolist()

    m_path = self.restrided_paths[idx]
    m_rots = self.rots[idx]
    m_trans = self.trans[idx]
    loaded_panos = self.load_and_resize_panos(m_path)[:, :, :, :3]
    sample = {"rgb_panos": loaded_panos, "rots": m_rots, "trans": m_trans}
    return sample
