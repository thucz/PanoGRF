import cv2
import os

from torchvision import transforms
from datasets.util import Equirec2Cube
# def preprocess():
def to_torch_data(image_path, height=512, width=1024):
    # max_depth_meters = 10.0
    e2c = Equirec2Cube(height, width, height // 2)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    rgb_name = os.path.join(image_path)
    rgb = cv2.imread(rgb_name)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

    # aug_rgb = rgb
    cube_rgb = e2c.run(rgb)
    rgb = to_tensor(rgb.copy())
    cube_rgb = to_tensor(cube_rgb.copy())

    inputs = {}
    inputs["rgb"] = rgb.unsqueeze(0)
    inputs["normalized_rgb"] = normalize(rgb).unsqueeze(0)
    inputs["cube_rgb"] = cube_rgb.unsqueeze(0)
    inputs["normalized_cube_rgb"] = normalize(cube_rgb).unsqueeze(0)
    return inputs
