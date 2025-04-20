import torch
import tensorflow as tf


def torch_version(input, cpu=True):
    # input
    x = torch.tensor(input["input"])

    if not cpu:
        x = x.cuda()

    # output
    y = torch.exp2(x)

    if not cpu:
        y = y.cpu()

    return {"exp2": y.numpy()}


def tensorflow_version(input, cpu=True):
    if cpu:
        device_string = "/cpu"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # pre-condition
        tf.experimental.numpy.experimental_enable_numpy_behavior()
        
        # input
        x = tf.constant(input["input"])

        # output
        y = tf.experimental.numpy.exp2(x)
        
        return {"exp2": y.numpy()}
