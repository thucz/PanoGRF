import os
import torch


def load_model(model, load_weights_dir):
    """
    Load model from disk
    """
    path = os.path.join(load_weights_dir, "{}.pth".format("model"))
    model_dict = model.state_dict()
    pretrained_dict = torch.load(path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model
