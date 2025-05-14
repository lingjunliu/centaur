import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    output_size = input_dict["output_size"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    adaptive_max_pool = nn.AdaptiveMaxPool2d(output_size)

    if not cpu:
        adaptive_max_pool = adaptive_max_pool.cuda()

    result = adaptive_max_pool(input_tensor)

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
        output_size = input_dict["output_size"]

        target_height, target_width = output_size

        result = tf.image.resize(
            input_tensor,
            [target_height, target_width],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )

        result = tf.nn.max_pool(
            tf.transpose(result, perm=[0, 2, 3, 1]),
            ksize=[1, 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        result = tf.transpose(result, perm=[0, 3, 1, 2])
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "output_size": (16, 16)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()