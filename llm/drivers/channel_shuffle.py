import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    groups = input_dict["groups"]
    dim = input_dict["dim"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.channel_shuffle(input_tensor, groups)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    groups = input_dict["groups"]
    dim = input_dict["dim"]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):

        shape = tf.shape(input_tensor)
        if dim < 0:
            dim = dim + len(input_tensor.shape)
        
        transposed = tf.transpose(
            tf.reshape(input_tensor,
                       tf.concat([shape[:dim], [groups, shape[dim] // groups], shape[dim + 1:]], axis=0)),
            perm=tf.concat([tf.range(dim, dtype=tf.int32), [dim + 1, dim], tf.range(dim + 2, len(input_tensor.shape) + 1, dtype=tf.int32)], axis=0)
        )
        result = tf.reshape(transposed, shape)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[[ 0.1131,  0.1435],
                              [ 0.7299, -0.4775],
                              [ 0.1166,  0.1607]],

                             [[ 0.8827, -0.5071],
                              [-0.6467, -0.8937],
                              [ 0.3190,  0.7359]],

                             [[-0.2356,  0.2369],
                              [ 0.5654, -0.1192],
                              [ 0.3583,  0.0947]],

                             [[ 1.3221, -1.6469],
                              [-0.5022,  0.0313],
                              [-0.5340,  0.9324]]]], dtype=np.float32),
        "groups": 2,
        "dim": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()