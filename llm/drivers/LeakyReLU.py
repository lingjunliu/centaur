import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    negative_slope = input_dict.get("negative_slope", 0.01)
    inplace = input_dict.get("inplace", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    leaky_relu = torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
    result = leaky_relu(input_tensor)

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
        negative_slope = input_dict.get("negative_slope", 0.01)

        result = tf.nn.leaky_relu(input_tensor, alpha=negative_slope)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32),
        "negative_slope": 0.1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()