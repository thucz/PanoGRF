# Lint as: python3
"""This data reader parses the CARLA dataset.
"""

import json
import os

import numpy as np
import pyproj
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from helpers import my_helpers
from helpers import my_torch_helpers


class CarlaReader(Dataset):
  """CARLA Dataset Reader"""

  def __init__(self,
               data_path,
               width=512,
               height=256,
               towns=("Town01",),
               min_dist=0,
               max_dist=None,
               seq_len=2,
               reference_idx=-1,
               use_euclidean_dist=True,
               use_meters_depth=False,
               interpolation_mode="bilinear",
               sampling_method="sparse",
               custom_params={},
               return_path = False,
               verbose=False):
    """Create a dataset reader from your CARLA dataset

    Args:
      data_path: Path to the dataset.
      width: width.
      height: height.
      towns: List of towns to read.
      min_dist: Minimum distance between panoramas in meters.
      seq_len: Length of sequences.
      reference_idx: Index of the reference pano for each sequence.
      use_euclidean_dist: Use euclidean distance to camera rather than z-depth.
      use_meters_depth: Scale depth from millimeters to meters.
      interpolation_mode: Either bilinear or nearest.
      sampling_method: Either sparse or dense.
        Use sparse for making GIFs, dense for training.
      verbose: Print debugging statements.
    """
    self.data_path = data_path
    self.width = width
    self.height = height
    self.towns = towns
    self.min_dist = min_dist
    self.max_dist = max_dist
    self.seq_len = seq_len
    self.reference_idx = reference_idx
    self.use_euclidean_dist = use_euclidean_dist
    self.use_meters_depth = use_meters_depth
    self.interpolation_mode = interpolation_mode
    self.sampling_method = sampling_method
    self.verbose = verbose
    self.return_path = return_path
    print("Carla dist: [%f, %f]" % (min_dist, max_dist if max_dist else min_dist))

    self.depth_to_dist_cache = {}
    self.geod = pyproj.Geod(ellps="WGS84")

    if sampling_method not in ['sparse', 'dense', 'custom']:
      raise ValueError('Invalid sampling method')

    if data_path != "":
      if sampling_method == "sparse":
        raw_paths, pano_metadata = self.do_sparse_sampling(towns=towns,
                                                          data_path=data_path,
                                                          min_dist=min_dist,
                                                          max_dist=max_dist)
        self.restrided_paths = self.generate_subsequences(raw_paths, seq_len)
        self.pano_metadata = pano_metadata
      elif sampling_method == "dense":
        restrided_paths, pano_metadata = self.do_dense_sampling(
          towns=towns,
          data_path=data_path,
          min_dist=min_dist,
          max_dist=max_dist,
          seq_len=seq_len)
        self.restrided_paths = restrided_paths
        self.pano_metadata = pano_metadata
      else:
        restrided_paths, pano_metadata = self.load_custom_sequence(
          town=custom_params['town'],
          run_id=custom_params['run_id'],
          data_path=data_path,
          start_frame=custom_params['start_frame'],
          distance=custom_params['distance'],
          seq_len=seq_len)
        self.restrided_paths = restrided_paths
        self.pano_metadata = pano_metadata
      if verbose:
        print("Sampling method", sampling_method)
        print("Restrided paths shape", self.restrided_paths.shape)

  def do_dense_sampling(self, towns, data_path, min_dist, max_dist, seq_len):
    """Performs a dense sampling of tuples where images are min_dist to max_dist
    apart.

    Each tuple will have seq_len images.

    Args:
      towns: List of towns.
      data_path: Path to the carla dataset.
      min_dist: Minimum distance between panos in each tuple.
      max_dist: Maximum distance between panos in each tuple.

    Returns:
      Array of image tuples shaped (N, seq_len).
      pano_metadata: Metadata of included panos.

    """
    restrided_paths = []
    pano_metadata = {}
    if max_dist is None:
      max_dist = min_dist
    min_dist_tiled = np.tile(min_dist, seq_len)
    max_dist_tiled = np.tile(max_dist, seq_len)
    # For each Town
    for i, town in enumerate(towns):
      print("Parsing Town %d/%d" % (i + 1, len(towns)))
      town_dir = os.path.join(data_path, town)
      town_paths = os.listdir(town_dir)
      # For each run
      for j, run_id in enumerate(town_paths):
        run_dir = os.path.join(town_dir, run_id)
        with open(os.path.join(run_dir, "gnss_data.json"), "r") as f:
          gnss_data = json.load(f)
        pano_indices = np.tile(0, seq_len)
        # Distances between panoramas.
        next_dist = min_dist_tiled + (
            max_dist_tiled - min_dist_tiled) * np.random.random(seq_len)
        while pano_indices[-1] < len(gnss_data):
          pano_idx = pano_indices[0]
          prev_pano_position = (gnss_data[pano_idx]["latitude"],
                                gnss_data[pano_idx]["longitude"])
          for k in range(1, seq_len):
            pano_idx = max(pano_indices[k], pano_indices[k - 1])
            my_lat = gnss_data[pano_idx]["latitude"]
            my_lng = gnss_data[pano_idx]["longitude"]
            azimuth, bazimuth, distance = self.geod.inv(my_lng, my_lat,
                                                        prev_pano_position[1],
                                                        prev_pano_position[0])
            while distance < next_dist[k] and pano_idx < len(gnss_data):
              pano_idx = pano_idx + 1
              while pano_idx < len(
                  gnss_data) and not self.pano_depth_file_exists(
                data_path, town, run_id, gnss_data[pano_idx]["frame"]):
                pano_idx = pano_idx + 1
              if pano_idx < len(gnss_data):
                my_lat = gnss_data[pano_idx]["latitude"]
                my_lng = gnss_data[pano_idx]["longitude"]
                azimuth, bazimuth, distance = self.geod.inv(
                  my_lng, my_lat, prev_pano_position[1],
                  prev_pano_position[0])
            # print("Distance", distance)
            if pano_idx >= len(gnss_data):
              # print("Pano_idx", pano_idx)
              pano_indices[-1] = pano_idx
              break
            pano_indices[k] = pano_idx
            prev_pano_position = (my_lat, my_lng)
          if pano_indices[-1] < len(gnss_data):
            # Save this current set of panos as a tuple.
            current_set = []
            save_set = True
            for k in range(seq_len):
              pano_idx = pano_indices[k]
              frame = gnss_data[pano_idx]["frame"]
              pano_id = ",".join((town, run_id, str(frame)))
              pano_metadata[pano_id] = gnss_data[pano_idx]
              current_set.append(pano_id)
              if not self.pano_depth_file_exists(data_path, town, run_id,
                                                 frame):
                save_set = False
                break
            if save_set:
              restrided_paths.append(current_set)
          # Increment the first index.
          pano_indices[0] = pano_indices[0] + 1
          while pano_indices[0] < len(gnss_data) and \
              not self.pano_depth_file_exists(
                data_path, town, run_id, gnss_data[pano_indices[0]]["frame"]):
            pano_indices[0] = pano_indices[0] + 1
          pano_indices[-1] = max(pano_indices[0], pano_indices[-1])
          # Set new random distances.
          next_dist = min_dist_tiled + (
              max_dist_tiled - min_dist_tiled) * np.random.random(seq_len)
    restrided_paths = np.array(restrided_paths)
    return restrided_paths, pano_metadata

  def do_sparse_sampling(self, towns, data_path, min_dist, max_dist):
    """Performs a sparse sampling of the data, creating paths where frames are
    between min_dist and max_dist meters apart.

    Args:
      towns: Array of towns.
      data_path: Data path.
      min_dist: Minimum distance between frames.
      max_dist: Maximum distance between frames.

    Returns:
      raw_paths: List of raw paths.
      pano_metadata: Metadata of all panos included.

    """
    pano_metadata = {}
    raw_paths = []
    for i, town in enumerate(towns):
      town_dir = os.path.join(data_path, town)
      town_paths = os.listdir(town_dir)
      for j, run_id in enumerate(town_paths):
        run_dir = os.path.join(town_dir, run_id)
        with open(os.path.join(run_dir, "gnss_data.json"), "r") as f:
          gnss_data = json.load(f)
        run_panos = []
        prev_pano_position = (90, 150)
        next_dist = min_dist
        if max_dist is not None:
          next_dist = min_dist + (max_dist - min_dist) * np.random.random()
        for k in range(len(gnss_data)):
          frame = gnss_data[k]["frame"]
          pano_id = ",".join((town, run_id, str(frame)))
          # print("Gnss data", gnss_data[k])
          my_lat = gnss_data[k]["latitude"]
          my_lng = gnss_data[k]["longitude"]
          azimuth, bazimuth, distance = self.geod.inv(my_lng, my_lat,
                                                      prev_pano_position[1],
                                                      prev_pano_position[0])
          pano_file = os.path.join(data_path, town, run_id, "rgb",
                                   str(frame) + ".png")
          depth_file = os.path.join(data_path, town, run_id, "depth",
                                    str(frame) + ".png")
          if (min_dist == 0 or distance >= next_dist) and \
              os.path.isfile(pano_file) and \
              os.path.isfile(depth_file):
            pano_metadata[pano_id] = gnss_data[k]
            run_panos.append(pano_id)
            prev_pano_position = (my_lat, my_lng)
            if max_dist is not None:
              next_dist = min_dist + (max_dist - min_dist) * np.random.random()
        raw_paths.append(run_panos)
    return raw_paths, pano_metadata

  def load_custom_sequence(self, town, run_id, start_frame, data_path, distance,
                           seq_len):
    pano_metadata = {}
    raw_paths = []
    town_dir = os.path.join(data_path, town)
    run_dir = os.path.join(town_dir, run_id)
    with open(os.path.join(run_dir, "gnss_data.json"), "r") as f:
      gnss_data = json.load(f)
    run_panos = []
    prev_pano_position = (90, 150)
    next_dist = distance
    for k in range(len(gnss_data)):
      frame = gnss_data[k]["frame"]
      if frame < start_frame or len(run_panos) >= seq_len:
        pass
      elif frame == start_frame:
        pano_id = ",".join((town, run_id, str(frame)))
        my_lat = gnss_data[k]["latitude"]
        my_lng = gnss_data[k]["longitude"]
        pano_file = os.path.join(data_path, town, run_id, "rgb",
                                 str(frame) + ".png")
        depth_file = os.path.join(data_path, town, run_id, "depth",
                                  str(frame) + ".png")
        if os.path.isfile(pano_file) and \
            os.path.isfile(depth_file):
          pano_metadata[pano_id] = gnss_data[k]
          run_panos.append(pano_id)
          prev_pano_position = (my_lat, my_lng)
        else:
          raise ValueError("Frame not found")
      else:
        pano_id = ",".join((town, run_id, str(frame)))
        # print("Gnss data", gnss_data[k])
        my_lat = gnss_data[k]["latitude"]
        my_lng = gnss_data[k]["longitude"]
        azimuth, bazimuth, distance = self.geod.inv(my_lng, my_lat,
                                                    prev_pano_position[1],
                                                    prev_pano_position[0])
        pano_file = os.path.join(data_path, town, run_id, "rgb",
                                 str(frame) + ".png")
        depth_file = os.path.join(data_path, town, run_id, "depth",
                                  str(frame) + ".png")
        if distance >= next_dist and \
            os.path.isfile(pano_file) and \
            os.path.isfile(depth_file):
          pano_metadata[pano_id] = gnss_data[k]
          run_panos.append(pano_id)
          prev_pano_position = (my_lat, my_lng)
    raw_paths.append(run_panos)
    raw_paths = np.array(raw_paths)
    return raw_paths, pano_metadata

  def pano_depth_file_exists(self, data_path, town, run_id, frame):
    pano_file = os.path.join(data_path, town, run_id, "rgb",
                             str(frame) + ".png")
    depth_file = os.path.join(data_path, town, run_id, "depth",
                              str(frame) + ".png")
    return os.path.isfile(pano_file) and os.path.isfile(depth_file)

  def generate_subsequences(self, sequences, seq_len=2):
    """Takes an array of sequences of arbitrary length and generates
    an array of sequences with fixed length.

    Args:
      sequences: array of sequences with arbitrary length
      seq_len: length of target sequences (Default value = 2)

    Returns:
      Array of sequence with length len.

    """
    new_paths = []
    for path in sequences:
      path = np.array(path)
      if len(path) >= seq_len:
        shape = (len(path) - seq_len + 1, seq_len)
        strides = (path.itemsize, path.itemsize)
        strided = np.lib.stride_tricks.as_strided(path,
                                                  shape=shape,
                                                  strides=strides)
        new_paths.append(strided)
    return np.concatenate(new_paths, axis=0)

  def load_and_resize_panos(self, pano_ids):
    """
    Loads and resizes panoramas as float32 tensors.

    Args:
      pano_ids: Array of panoids.

    Returns: Resized rgb panos and depth panos stacked along 0 axis.

    """
    rgb_panos = []
    depth_panos = []
    for i, pano_id in enumerate(pano_ids):
      town, run_id, frame = pano_id.split(",")
      rgb_pano_path = os.path.join(self.data_path, town, run_id, "rgb",
                                   frame + ".png")
      rgb_img = Image.open(rgb_pano_path).resize((self.width, self.height))
      rgb_img = np.array(rgb_img, dtype=np.float32) / 255
      rgb_panos.append(rgb_img)
      depth_pano_path = os.path.join(self.data_path, town, run_id, "depth",
                                     frame + ".png")
      depth_img = Image.open(depth_pano_path)
      depth_img = np.array(depth_img, dtype=np.float32)
      depth_panos.append(depth_img)
    rgb_panos = np.stack(rgb_panos, axis=0)
    depth_panos = np.stack(depth_panos, axis=0)
    if self.use_euclidean_dist:
      depth_panos = self.zdepth_to_distance(depth_panos[:, :, :, np.newaxis])
      depth_panos = depth_panos[:, :, :, 0].astype(np.float32)
    if self.use_meters_depth:
      depth_panos = 0.001 * depth_panos
    return rgb_panos, depth_panos

  def zdepth_to_distance(self, depth_image):
    """Converts a depth (z-depth) image to a euclidean distance image.

    Args:
      depth_image: Equirectangular depth image as BxHxWx1 array.

    Returns: Equirectangular distance image.

    """
    batch_size, height, width, channels = depth_image.shape
    cache_key = "_".join((str(height), str(width)))
    self.cache_depth_to_dist(height, width)
    ratio = self.depth_to_dist_cache[cache_key]
    new_depth_image = depth_image * ratio[np.newaxis, :, :, np.newaxis]
    return new_depth_image

  def zdepth_to_distance_torch(self, depth_image):
    """Converts a depth (z-depth) image to a euclidean distance image.

    Args:
      depth_image: Equirectangular depth image as BxHxWx1 tensor.

    Returns: Equirectangular distance image.

    """
    batch_size, height, width, channels = depth_image.shape
    cache_key = "_".join((str(height), str(width)))
    self.cache_depth_to_dist(height, width)
    ratio = self.depth_to_dist_cache[cache_key]
    ratio = ratio[np.newaxis, :, :, np.newaxis]
    ratio = torch.tensor(ratio,
                         dtype=depth_image.dtype,
                         device=depth_image.device)
    new_depth_image = depth_image * ratio
    return new_depth_image

  def distance_to_zdepth_torch(self, distance_image):
    """Converts a euclidean depth to z-depth image.

    Args:
      depth_image: Equirectangular depth image as BxHxWx1 tensor.

    Returns: Equirectangular distance image.

    """
    batch_size, height, width, channels = distance_image.shape
    cache_key = "_".join((str(height), str(width)))
    self.cache_depth_to_dist(height, width)
    ratio = self.depth_to_dist_cache[cache_key]
    ratio = ratio[np.newaxis, :, :, np.newaxis]
    ratio = torch.tensor(ratio,
                         dtype=distance_image.dtype,
                         device=distance_image.device)
    zdepth_image = my_torch_helpers.safe_divide(distance_image, ratio)
    return zdepth_image

  def cache_depth_to_dist(self, height, width):
    """Caches a depth to dist ratio"""
    cache_key = "_".join((str(height), str(width)))
    if cache_key not in self.depth_to_dist_cache:
      cubemap_height = 256
      cubemap_width = 256
      # Distance to image plane
      theta, phi = np.meshgrid((np.arange(width) + 0.5) * (2 * np.pi / width),
                               (np.arange(height) + 0.5) * (np.pi / height))
      uvs, uv_sides = my_helpers.spherical_to_cubemap(theta.reshape(-1),
                                                      phi.reshape(-1))
      cubemap_uvs = uvs.reshape(height, width, 2)
      uv_int = np.stack(
        (cubemap_uvs[:, :, 0] * (cubemap_width - 1), cubemap_uvs[:, :, 1] *
         (cubemap_height - 1)),
        axis=-1)
      width_center = cubemap_width / 2 - 0.5
      height_center = cubemap_height / 2 - 0.5
      focal_len = (cubemap_height / 2) / np.tan(np.pi / 4)
      diag_dist = np.sqrt((uv_int[:, :, 0] - width_center) ** 2 +
                          (uv_int[:, :,
                           1] - height_center) ** 2 + focal_len ** 2)
      self.depth_to_dist_cache[cache_key] = diag_dist / focal_len

  def parse_rotation(self, rotation):
    """Parses carla's rotation to our coordinate system.

    Args:
      rotation: rotation dict

    Returns: scipy.spatial.transforms.Rotation instance

    """
    # rot = [-rotation["pitch"], rotation["yaw"], rotation["roll"]]
    # return Rotation.from_euler('xyz', rot, degrees=True)
    rot = [-rotation["roll"], -rotation["yaw"], rotation["pitch"]]
    return Rotation.from_euler('zyx', rot, degrees=True)

  def parse_location(self, location, dtype=np.float32):
    """Converts carla exported location to our coordinate system.

    Args:
      location: location dict
      dtype: dtype (default = float32)

    Returns: np.array

    """
    return np.array([location["y"], location["z"], -location["x"]], dtype=dtype)

  def calculate_rot_trans(self, paths, reference_idx=-1):
    """Calculates rotation and translation of each pano in the path
    relative to a single pano.

    Args:
      paths: Array of pano ids of size BxP.
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
      reference_lat = reference_metadata["latitude"]
      reference_lng = reference_metadata["longitude"]
      reference_rot = self.parse_rotation(reference_metadata["rotation"])
      # print("Reference_rotation", reference_metadata["rotation"])
      reference_rot_inv = reference_rot.inv()
      reference_location = self.parse_location(reference_metadata["location"])
      for j, pano_id in enumerate(my_path):
        if j is reference_idx:
          rot = np.eye(3, dtype=np.float32)
          t = np.zeros(3, dtype=np.float32)
        else:
          my_metadata = self.pano_metadata[pano_id]
          my_lat = my_metadata["latitude"]
          my_lng = my_metadata["longitude"]
          my_rot = self.parse_rotation(my_metadata["rotation"])
          # print("My rotation", j, my_metadata["rotation"])
          my_rot_inv = my_rot.inv()
          my_location = self.parse_location(my_metadata["location"])
          location_delta = my_location - reference_location
          location_delta = -reference_rot_inv.apply(location_delta)
          _, _, distance = self.geod.inv(my_lng, my_lat, reference_lng,
                                         reference_lat)
          rot = (reference_rot * my_rot_inv).as_matrix()
          t = location_delta
          if distance > 0:
            t = distance * location_delta / np.linalg.norm(location_delta)
          # print("pano_id", pano_id, rot.shape)
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
      Dictionary including rgb_panos, rotations, and translations,
      and depth_panos.
    """
    if torch.is_tensor(idx):
      idx = idx.tolist()

    m_path = self.restrided_paths[idx]
    m_rots, m_trans = self.calculate_rot_trans(m_path[np.newaxis, ...],
                                               reference_idx=self.reference_idx)
    m_rots = m_rots[0]
    m_trans = m_trans[0]
    rgb_panos, depth_panos = self.load_and_resize_panos(m_path)
    sample = {
      "rgb_panos": rgb_panos[:, :, :, :3],
      "rots": m_rots,
      "trans": m_trans,
      "depth_panos": depth_panos
    }
    if self.return_path:
      sample["path"] = m_path
    return sample
