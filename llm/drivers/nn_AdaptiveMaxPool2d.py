import numpy as np
import tensorflow as tf

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    output_size = input_dict["output_size"]
    return_indices = input_dict.get("return_indices", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    m = torch.nn.AdaptiveMaxPool2d(output_size, return_indices=return_indices)
    result = m(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    input_np = input_dict["input"]
    output_size = input_dict["output_size"]

    if isinstance(output_size, int):
        target_height = output_size
        target_width = output_size
    elif isinstance(output_size, tuple):
        target_height = output_size[0] if output_size[0] is not None else input_np.shape[-2]
        target_width = output_size[1] if output_size[1] is not None else input_np.shape[-1]
    else:
        raise ValueError("output_size must be an int or tuple")

    input_tensor = tf.convert_to_tensor(input_np, dtype=tf.float32)
    if len(input_np.shape) == 3:
        input_tensor = tf.expand_dims(input_tensor, axis=0)

    input_shape = tf.shape(input_tensor)
    batch_size = input_shape[0]
    num_channels = input_shape[1]
    input_height = input_shape[2]
    input_width = input_shape[3]

    def adaptive_max_pool2d(input_tensor, output_size):
        output_height, output_width = output_size
        in_height, in_width = input_tensor.shape[1], input_tensor.shape[2]

        def calculate_start_end(in_size, out_size):
            stride = in_size // out_size
            kernel_size = in_size - (out_size - 1) * stride
            start_indices = tf.range(0, in_size, stride)
            end_indices = start_indices + kernel_size
            end_indices = tf.clip_by_value(end_indices, clip_value_min=0, clip_value_max=in_size)
            return start_indices, end_indices

        start_h, end_h = calculate_start_end(in_height, output_height)
        start_w, end_w = calculate_start_end(in_width, output_width)

        output = tf.TensorArray(tf.float32, size=output_height)
        for i in range(output_height):
            row = tf.TensorArray(tf.float32, size=output_width)
            for j in range(output_width):
                h_start, h_end = start_h[i], end_h[i]
                w_start, w_end = start_w[j], end_w[j]
                patch = input_tensor[:, h_start:h_end, w_start:w_end, :]
                max_val = tf.reduce_max(patch, axis=[1, 2], keepdims=False)
                row = row.write(j, max_val)
            output = output.write(i, row.stack())

        output = output.stack()
        if len(input_np.shape) == 3:
            output = tf.transpose(output, perm=[0, 1, 2])
        else:
            output = tf.transpose(output, perm=[1, 0, 2, 3], )

        return output

    result = adaptive_max_pool2d(input_tensor, (target_height, target_width))

    if len(input_np.shape) == 3:
        result = tf.squeeze(result, axis=0)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.randn(1, 64, 8, 9).astype(np.float32),
        "output_size": (5, 7)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    tf_result_reshaped = np.transpose(tf_result["result"], (0, 1, 2,3))

    assert np.allclose(torch_result["result"], tf_result_reshaped, atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.randn(1, 64, 10, 9).astype(np.float32),
        "output_size": 7
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    tf_result_reshaped = np.transpose(tf_result["result"], (0, 1, 2,3))


    assert np.allclose(torch_result["result"], tf_result_reshaped, atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.random.randn(1, 64, 10, 9).astype(np.float32),
        "output_size": (None, 7)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    tf_result_reshaped = np.transpose(tf_result["result"], (0, 1, 2,3))


    assert np.allclose(torch_result["result"], tf_result_reshaped, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()