import torch
import tensorflow as tf
from collections.abc import Iterable


def torch_version(input, cpu=True):
    # input
    x1 = torch.tensor(input["input"])
    if "dim" in input:
        x2 = input["dim"]

    if "keepdims" in input:
        keepdims = input["keepdims"]
    else:
        keepdims = False

    if "correction" in input:
        correction = input["correction"]
    else:
        correction = 0
    
    if not cpu:
        x1 = x1.cuda()

    # output
    if "dim" in input:
        y = torch.std_mean(x1, x2, keepdims=keepdims, correction=correction)
    else:
        y = torch.std_mean(x1)
    
    if not cpu:
        y = (y[0].cpu(), y[1].cpu())

    return {"std": y[0].numpy(), "mean": y[1].numpy()}


def tensorflow_version(input, cpu=True):
    if cpu:
        device_string = "/cpu"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # input
        x1 = tf.constant(input["input"])
        
        if "dim" in input:
            if isinstance(input["dim"], Iterable):
                x2 = list(input["dim"])
            else:
                x2 = [input["dim"]]
        
        if "keepdims" in input:
            keepdims = input["keepdims"]
        else:
            keepdims = False
        
        # output
        if "dim" in input:
            y = tf.nn.moments(x1, x2, keepdims=keepdims)
        else:
            y = tf.nn.moments(x1)
        
        return {"std": tf.sqrt(y[1]).numpy(), "mean": y[0].numpy()}