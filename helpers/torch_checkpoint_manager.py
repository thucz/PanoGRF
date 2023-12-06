# Lint as: python3
"""Basic checkpoint manager for pytorch.
"""
import torch
import os
import sys
import re


class CheckpointManager:
  """Basic checkpoint manager which saves and loads checkpoints. It also keeps track of the current step.
  """

  def __init__(self, checkpoint_dir, max_to_keep=3, device="cuda", name=""):
    """Create a checkpoint manager.

    Args:
      checkpoint_dir: Path to checkpoint directory.
    """
    self.checkpoint_dir = checkpoint_dir
    self.step = 0
    self.reference_path = os.path.join(self.checkpoint_dir, "checkpoint")
    self.max_to_keep = max_to_keep
    self.device = device
    self.name = name
    self.frozen = False
    os.makedirs(checkpoint_dir, exist_ok=True)

  def load_latest_checkpoint(self):
    reference_path = self.reference_path
    checkpoint_file = None
    if os.path.isfile(reference_path):
      with open(reference_path, "r") as f:
        checkpoint_file = f.readline()
    if checkpoint_file is not None:
      self.step = int(re.match(r"checkpoint_(\d+)\.pt", checkpoint_file)[1])
      return torch.load(os.path.join(self.checkpoint_dir, checkpoint_file), map_location=self.device)
    return None

  def save_checkpoint(self, obj):
    """Saves a checkpoint with the current step."""
    assert not self.frozen, "Checkpoint manager frozen"
    checkpoint_filename = "checkpoint_%d.pt" % self.step
    save_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
    torch.save(obj, save_path)
    reference_path = self.reference_path
    with open(reference_path, "w") as f:
      f.write(checkpoint_filename)
    # self.clean_checkpoints()

  # def clean_checkpoints(self):
  #   """Deletes checkpoints older than max_to_keep."""
  #   assert not self.frozen, "Checkpoint manager frozen"
  #   all_checkpoints = os.listdir(self.checkpoint_dir)
  #   numbered_checkpoints = []
  #   for ckpt in all_checkpoints:
  #     match = re.match(r"checkpoint_(\d+)\.pt", ckpt)
  #     if match is not None:
  #       numbered_checkpoints.append({"filename": ckpt, "num": int(match[1])})
  #   numbered_checkpoints.sort(key=lambda x: x["num"])
  #   for ckpt in numbered_checkpoints[:-self.max_to_keep]:
  #     checkpoint_path = os.path.join(self.checkpoint_dir, ckpt["filename"])
  #     os.remove(checkpoint_path)

  def increment_step(self):
    """Increments step and returns the new step

    Returns:New step

    """
    assert not self.frozen, "Checkpoint manager frozen"
    self.step = self.step + 1
    return self.step

  def freeze(self):
    """Freeze the checkpoint manager. Disables all functionality of the checkpoint manager except loading.

    Returns: self.

    """
    self.frozen = True
    return self