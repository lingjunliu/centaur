import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    size = input_dict.get("size", 5)
    alpha = input_dict.get("alpha", 0.0001)
    beta = input_dict.get("beta", 0.75)
    k = input_dict.get("k", 1.0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    lrn = torch.nn.LocalResponseNorm(size, alpha, beta, k)
    result = lrn(input_tensor)

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
        size = input_dict.get("size", 5)
        alpha = input_dict.get("alpha", 0.0001)
        beta = input_dict.get("beta", 0.75)
        k = input_dict.get("k", 1.0)

        input_shape = tf.shape(input_tensor)
        ndims = len(input_tensor.shape)
        if ndims == 2:
            input_tensor = tf.expand_dims(input_tensor, axis=0)
            input_tensor = tf.expand_dims(input_tensor, axis=-1)
        elif ndims == 3:
            input_tensor = tf.expand_dims(input_tensor, axis=0)
        
        radius = (size - 1) // 2
        square_sum = tf.nn.local_response_normalization(
            input_tensor,
            depth_radius=radius,
            bias=k,
            alpha=alpha,
            beta=beta
        )

        if ndims == 2:
            square_sum = tf.squeeze(square_sum, axis=[0, -1])
        elif ndims == 3:
            square_sum = tf.squeeze(square_sum, axis=0)

        result = square_sum.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[[0.0202, 1.0985, 1.3506, -0.6056],
                              [0.0202, 1.0985, 1.3506, -0.6056],
                              [0.0202, 1.0985, 1.3506, -0.6056]]]], dtype=np.float32),
        "size": 5,
        "alpha": 0.0001,
        "beta": 0.75,
        "k": 1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()