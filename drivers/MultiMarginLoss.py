import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

# TODO: Add other parameters
def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    target_tensor = torch.tensor(input["target"])
    m = torch.nn.MultiMarginLoss()

    if not cpu:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()
        m = m.cuda()
    
    result = m(input_tensor, target_tensor)

    if not cpu:
        result = result.cpu()

    return {"MultiMarginLoss": result.numpy()}

# TODO: Write TensorFlow version of the function