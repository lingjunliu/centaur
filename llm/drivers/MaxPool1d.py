import numpy as np
import tensorflow as tf

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", None)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    return_indices = input_dict.get("return_indices", False)
    ceil_mode = input_dict.get("ceil_mode", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    m = torch.nn.MaxPool1d(kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)
    result = m(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.squeeze().numpy()}

def tensorflow_version(input_dict, cpu=True):

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        kernel_size = input_dict["kernel_size"]
        stride = input_dict.get("stride", kernel_size)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        ceil_mode = input_dict.get("ceil_mode", False)

        input_tensor = tf.expand_dims(input_tensor, axis=0)
        input_tensor = tf.expand_dims(input_tensor, axis=-1)

        if padding > 0:
            paddings = [[0, 0], [padding, padding]]
            input_tensor = tf.pad(input_tensor, paddings, "CONSTANT", constant_values=-float('inf'))
        
        if dilation > 1:
            ksize_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
            
            patches = tf.extract_volume_patches(
                input=tf.expand_dims(input_tensor, axis=0),
                ksizes=[1, 1, ksize_effective, 1, 1],
                strides=[1, 1, stride, 1, 1],
                padding='VALID'
            )
            
            result = tf.reduce_max(patches, axis=[2, 4])
        else:
            result = tf.nn.pool(
                input=tf.expand_dims(input_tensor, axis=0),
                window_shape=[kernel_size],
                pooling_type='MAX',
                padding='VALID',
                strides=[stride]
            )

        if ceil_mode:
            input_len = input_dict["input"].shape[-1] + 2 * padding
            output_len = (input_len - (kernel_size - 1) - 1 + stride) // stride
            target_output_len = (input_len + stride - 1) // stride
            
            if output_len < target_output_len:
                result = result[:, :, :target_output_len, :]

        result = tf.squeeze(result).numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32),
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "ceil_mode": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32),
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "ceil_mode": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32),
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "ceil_mode": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32),
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "dilation": 2,
        "ceil_mode": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()