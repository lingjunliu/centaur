from numpy import dtype


def torch_version(input, cpu=True):
    import torch
    # input
    x1 = input["high"]
    x2 = input["size"]
    x3 = input["seed"]

    # output
    torch.manual_seed(seed=x3)
    y = torch.randint(x1, size=x2)

    if not cpu:
        y = y.cpu()

    return {"randint": y.numpy()}


def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    if cpu:
        device_string = "/cpu"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # pre-condition
        tf.experimental.numpy.experimental_enable_numpy_behavior()
        
        # input
        x1 = input["high"]
        x2 = input["size"]
        x3 = input["seed"]

        # output
        tf.experimental.numpy.random.seed(x3)
        y = tf.experimental.numpy.random.randint(low=0,high=x1, size=x2)
        
        return {"randint": y.numpy()}
