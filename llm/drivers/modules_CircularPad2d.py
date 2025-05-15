import numpy as np
import tensorflow as tf

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict.get("padding")

    if not cpu:
        input_tensor = input_tensor.cuda()

    pad = torch.nn.CircularPad2d(padding)
    result = pad(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):

    input_tensor = tf.constant(input_dict["input"])
    padding = input_dict.get("padding")

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        pad_left, pad_right, pad_top, pad_bottom = padding
        input_shape = input_tensor.shape
        batch_size, height, width, channels = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

        padded_height = height + pad_top + pad_bottom
        padded_width = width + pad_left + pad_right

        padded_tensor = np.zeros((batch_size, padded_height, padded_width, channels), dtype=input_tensor.dtype.as_numpy_dtype)

        input_np = input_tensor.numpy()
        for b in range(batch_size):
            for c in range(channels):
                for i in range(padded_height):
                    for j in range(padded_width):
                        y = (i - pad_top) % height
                        x = (j - pad_left) % width
                        padded_tensor[b, i, j, c] = input_np[b, y, x, c]

        result = padded_tensor

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 4, 2).astype(np.float32),
        "padding": (2, 2, 1, 1)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    torch_output_shape = torch_result['result'].shape
    tf_output_shape = tf_result['result'].shape
    if torch_output_shape != tf_output_shape:
        print(f"Torch output shape: {torch_output_shape}")
        print(f"TensorFlow output shape: {tf_output_shape}")

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()