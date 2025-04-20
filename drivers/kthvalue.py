# TODO: Fix



def torch_version(input, cpu=True):
    import torch
    # input
    x1 = torch.tensor(input["input"])
    x2 = input["k"]

    if not cpu:
        x1 = x1.cuda()

    # output
    y = torch.kthvalue(x1, x2)

    y_values = y.values
    y_indices = y.indices
    
    if not cpu:
        y_values = y_values.cpu()
        y_indices = y_indices.cpu()

    return {"values": y_values.numpy(), "indices": y_indices.numpy()}


def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    if cpu:
        device_string = "/cpu"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # input
        x1 = tf.constant(input["input"])
        x2 = input["k"]
        last_dim = x1.shape[-1]

        # output
        y = tf.math.top_k(x1, last_dim-x2, sorted=True)
        
        return {"values": y[0][:x2].numpy(), "indices": y[1][:x2].numpy()}
