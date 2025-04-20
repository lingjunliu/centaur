import torch
import tensorflow as tf


def torch_version(input, cpu=True):
    # input
    x = torch.tensor(input["input"])

    if not cpu:
        x = x.cuda()

    # output
    y = torch.digamma(x)

    if not cpu:
        y = y.cpu()

    return {"digamma": y.numpy()}


def tensorflow_version(input, cpu=True):
    if cpu:
        device_string = "/cpu"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # input
        x = tf.constant(input["input"])

        # output
        y = tf.math.digamma(x)
        
        return {"digamma": y.numpy()}
