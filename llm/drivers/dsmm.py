import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input = torch.tensor(input_dict["input"])
    mat2 = torch.tensor(input_dict["mat2"])

    if not cpu:
        input = input.cuda()
        mat2 = mat2.cuda()

    input_shape = input.shape
    mat2_shape = mat2.shape

    batch_dims = input_shape[:-2]
    m = input_shape[-2]
    k = input_shape[-1]
    n = mat2_shape[-1]

    input = input.reshape((-1, k))
    mat2 = mat2.reshape((k, n))

    result = torch.matmul(input, mat2)
    result = result.reshape(*batch_dims, m, n)

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
        input = tf.constant(input_dict["input"])
        mat2 = tf.constant(input_dict["mat2"])
        
        input_shape = tf.shape(input)
        mat2_shape = tf.shape(mat2)

        batch_dims = input_shape[:-2]
        m = input_shape[-2]
        k = input_shape[-1]
        n = mat2_shape[-1]

        input = tf.reshape(input, tf.concat([[-1], [k]], axis=0))
        mat2 = tf.reshape(mat2, tf.concat([[k, n]], axis=0))

        result = tf.matmul(input, mat2)
        result = tf.reshape(result, tf.concat([batch_dims, [m, n]], axis=0))

        result = result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 4).astype(np.float32),
        "mat2": np.random.rand(4, 5).astype(np.float32),
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(
        torch_result["result"], tf_result["result"], atol=A_TOL
    ), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()