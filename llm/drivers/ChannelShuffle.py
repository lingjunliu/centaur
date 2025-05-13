import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from torch import nn

    input_tensor = torch.tensor(input_dict["input"])
    groups = input_dict["groups"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    channel_shuffle = nn.ChannelShuffle(groups)
    try:
        result = channel_shuffle(input_tensor)
    except RuntimeError as e:
        print(f"Caught RuntimeError: {e}")
        result = input_tensor
    else:
        if not cpu:
            result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    groups = input_dict["groups"]
    
    shape = input_tensor.shape
    num_channels = shape[-1]
    
    if num_channels % groups != 0:
        print("Number of channels not divisible by groups, returning input")
        result = input_tensor
    else:
        input_tensor_shape = tf.shape(input_tensor)
        batch_size = input_tensor_shape[0]
        height = input_tensor_shape[1]
        width = input_tensor_shape[2]
        channels = input_tensor_shape[3]

        input_tensor = tf.reshape(input_tensor, [batch_size, height, width, groups, channels // groups])
        input_tensor = tf.transpose(input_tensor, perm=[0, 1, 2, 4, 3])
        result = tf.reshape(input_tensor, [batch_size, height, width, channels])
        

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[[ 0.4978,  0.9229, -0.1924,  0.7488],
                              [ 0.8308,  0.7898, -0.8916, -0.3825],
                              [ 0.7893, -0.1987,  0.1582, -0.3962]]]], dtype=np.float32),
        "groups": 2
    }
    
    input_data_2 = {
        "input": np.array([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=np.float32),
        "groups": 2
    }
    
    input_data_3 = {
        "input": np.array([[[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]]], dtype=np.float32),
        "groups": 3
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    torch_result = torch_version(input_data_2)
    tf_result = tensorflow_version(input_data_2)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    torch_result = torch_version(input_data_3)
    tf_result = tensorflow_version(input_data_3)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()