import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    input_tensor = torch.tensor(input_dict["input"], dtype=torch.float32)
    size = input_dict.get("size", 5)
    alpha = input_dict.get("alpha", 0.0001)
    beta = input_dict.get("beta", 0.75)
    k = input_dict.get("k", 1.0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    lrn = nn.CrossMapLRN2d(size, alpha, beta, k)

    if not cpu:
        lrn = lrn.cuda()

    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    result = lrn(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.squeeze().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    size = input_dict.get("size", 5)
    alpha = input_dict.get("alpha", 0.0001)
    beta = input_dict.get("beta", 0.75)
    k = input_dict.get("k", 1.0)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.reshape(input_tensor, [1, input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2]])

        def lrn(x, depth_radius, alpha, beta, bias=1.0):
            square_sum = tf.nn.local_response_normalization(x, depth_radius=depth_radius, alpha=alpha, beta=beta, bias=bias)
            return square_sum

        result = lrn(input_tensor, depth_radius=size // 2, alpha=alpha, beta=beta, bias=k)

        result = result.numpy()

    return {"result": result.squeeze()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], dtype=np.float32),
        "size": 3,
        "alpha": 0.0001,
        "beta": 0.75,
        "k": 2.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()