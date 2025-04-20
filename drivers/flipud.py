

def torch_version(input, cpu=True):
    import torch
    # input
    x = torch.tensor(input["input"])

    # output
    y = torch.flipud(x)

    if not cpu:
        y = y.cpu()

    return {"flipud": y.numpy()}


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
        y = tf.experimental.numpy.flipud(x)
        
        return {"flipud": y.numpy()}
