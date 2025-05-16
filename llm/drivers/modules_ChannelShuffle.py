import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from torch import nn

    input_tensor = torch.tensor(input_dict["input"])
    groups = input_dict["groups"]
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    channel_shuffle = nn.ChannelShuffle(groups)
    result = channel_shuffle(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        groups = input_dict["groups"]

        shape = tf.shape(input_tensor)
        batchsize = shape[0]
        num_channels = shape[1]
        height = shape[2]
        width = shape[3]

        channels_per_group = num_channels // groups

        x = tf.reshape(input_tensor, [batchsize, groups, channels_per_group, height, width])
        x = tf.transpose(x, [0, 2, 1, 3, 4])

        result = tf.reshape(x, [batchsize, num_channels, height, width])
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_shape = (1, 4, 2, 2)
    groups = 2
    input_data = {
        "input": np.random.rand(*input_shape).astype(np.float32),
        "groups": groups
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()