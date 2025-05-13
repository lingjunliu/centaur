import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    groups = input_dict["groups"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    batchsize, num_channels, height, width = input_tensor.data.size()
    channels_per_group = num_channels // groups

    input_tensor = input_tensor.view(batchsize, groups,
        channels_per_group, height, width)
    input_tensor = torch.transpose(input_tensor, 1, 2).contiguous()
    result = input_tensor.view(batchsize, -1, height, width)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        groups = input_dict["groups"]

        shape = input_tensor.shape
        batchsize, num_channels, height, width = shape[0], shape[1], shape[2], shape[3]
        channels_per_group = num_channels // groups

        input_tensor = tf.reshape(input_tensor, [batchsize, groups, channels_per_group, height, width])
        input_tensor = tf.transpose(input_tensor, perm=[0, 2, 1, 3, 4])
        result = tf.reshape(input_tensor, [batchsize, -1, height, width])

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 24, 32, 32).astype(np.float32),
        "groups": 3
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()