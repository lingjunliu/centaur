

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    x = torch.tensor(input["input"])
    
    if not cpu:
        x = x.cuda()

    y = torch.exp2(x)

    if not cpu:
        y = y.cpu()

    return {"exp2": y.numpy()}


def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    if cpu:
        device_string = "/cpu"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        x = tf.constant(input["input"])

        y = tf.math.pow(2, x)

        return {"exp2": y.numpy()}
