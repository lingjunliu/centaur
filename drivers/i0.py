

def torch_version(input, cpu=True):
    import torch
    # input
    x = torch.tensor(input["input"])

    if not cpu:
        x = x.cuda()

    # output
    y = torch.i0(x)

    if not cpu:
        y = y.cpu()

    return {"i0": y.numpy()}


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
        y = tf.math.bessel_i0(x)
        
        return {"i0": y.numpy()}
