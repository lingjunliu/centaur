import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict.get("padding", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    pad = torch.nn.CircularPad2d(padding)
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    result = pad(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.squeeze().numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    padding = input_dict.get("padding", 0)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_shape = tf.shape(input_tensor)
        height = input_shape[0]
        width = input_shape[1]

        if padding > 0:
            top_pad = input_tensor[-padding:, :]
            bottom_pad = input_tensor[:padding, :]
            padded_height = tf.concat([top_pad, input_tensor, bottom_pad], axis=0)

            left_pad = padded_height[:, -padding:]
            right_pad = padded_height[:, :padding]
            result = tf.concat([left_pad, padded_height, right_pad], axis=1)
        else:
            result = input_tensor
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(5, 5).astype(np.float32),
        "padding": 2,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()