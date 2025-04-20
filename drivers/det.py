import numpy as np


def torch_version(input, cpu=True):
    import torch
    # input
    if "A" in input:
        x = torch.tensor(input["A"])
    else:
        x = torch.tensor(input["input"])

    if not cpu:
        x = x.cuda()

    # output
    y = torch.det(x)

    if not cpu:
        y = y.cpu()

    if y.dim() == 0:
        y = np.float32(y.item())
    else:
        y = y.numpy()
    
    return {"det": y}


def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    if cpu:
        device_string = "/cpu"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # input
        if "A" in input:
            x = torch.tensor(input["A"])
        else:
            x = torch.tensor(input["input"])

        # output
        y = tf.linalg.det(x)
        
        return {"det": y.numpy()}
