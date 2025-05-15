import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None
    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    result = torch.nn.functional.linear(input_tensor, weight, bias)

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
        weight = tf.constant(input_dict["weight"])
        bias = tf.constant(input_dict["bias"]) if "bias" in input_dict else None

        result = tf.linalg.matmul(tf.expand_dims(input_tensor, axis=0), weight, transpose_b=True)
        if bias is not None:
            result = tf.add(result, bias)

        result = result.numpy().squeeze(axis=0)

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "weight": np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32),
        "bias": np.array([10.0, 11.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "weight": np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()