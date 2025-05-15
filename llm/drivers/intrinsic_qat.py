import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    scale = torch.tensor(input_dict["scale"])
    zero_point = torch.tensor(input_dict["zero_point"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        scale = scale.cuda()
        zero_point = zero_point.cuda()

    input_tensor_float = (input_tensor.float() - zero_point) * scale
    result = torch.nn.ReLU()(input_tensor_float)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        scale = tf.constant(input_dict["scale"], dtype=tf.float32)
        zero_point = tf.constant(input_dict["zero_point"], dtype=tf.float32)

        dequantized = (input_tensor - zero_point) * scale
        relu = tf.nn.relu(dequantized)

        result = relu.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, -2, 3, -4, 5], dtype=np.int8),
        "scale": np.array(0.5, dtype=np.float32),
        "zero_point": np.array(0, dtype=np.int32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()