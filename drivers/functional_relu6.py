from numpy import dtype
import torch
import tensorflow as tf


def torch_version(input, cpu=True):
    # input
    x = torch.tensor(input["input"])

    # output
    y = torch.nn.functional.relu6(x)

    if not cpu:
        y = y.cpu()

    return {"relu6": y.numpy()}


def tensorflow_version(input, cpu=True):
    if cpu:
        device_string = "/cpu"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # input
        x = tf.constant(input["input"])

        # output        
        y = tf.keras.activations.relu6(x)
        
        return {"relu6": y.numpy()}