import torch
import tensorflow as tf


def torch_version(input, cpu=True):
    # input
    x1 = torch.tensor(input["input"])
    x2 = torch.tensor(input["other"])

    if not cpu:
        x1 = x1.cuda()
        x2 = x2.cuda()

    # output
    y = torch.le(x1, x2)

    if not cpu:
        y = y.cpu()

    return {"le": y.numpy()}


def tensorflow_version(input, cpu=True):
    if cpu:
        device_string = "/cpu"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # input
        x1 = tf.constant(input["input"])
        x2 = tf.constant(input["other"])

        # output
        y = tf.less_equal(x1, x2)

        return {"le": y.numpy()}
