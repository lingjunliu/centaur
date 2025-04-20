from numpy import dtype


def torch_version(input, cpu=True):
    import torch
    # input
    x = torch.tensor(input["input"])

    # output
    y = torch.nn.functional.relu6(x)

    if not cpu:
        y = y.cpu()

    return {"relu6": y.numpy()}


def tensorflow_version(input, cpu=True):
    import tensorflow as tf
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