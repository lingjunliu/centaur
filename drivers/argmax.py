import torch
import tensorflow as tf


def torch_version(input, cpu=True):
    # input
    x1 = torch.tensor(input["input"])
    
    if "dim" in input:
        x2 = input["dim"]
    
    if not cpu:
        x1 = x1.cuda()

    # output
    if "dim" in input:
        y = torch.argmax(x1, dim=x2)
    else:
        y = torch.argmax(x1)

    if not cpu:
        y = y.cpu()

    return {"argmax": y.numpy()}


def tensorflow_version(input, cpu=True):
    if cpu:
        device_string = "/cpu"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # input
        x1 = tf.constant(input["input"])
        x2 = input.get("dim", None)

        # output        
        y = tf.argmax(x1, axis=x2)
        
        return {"argmax": y.numpy()}
