import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", None)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    return_indices = input_dict.get("return_indices", False)
    ceil_mode = input_dict.get("ceil_mode", False)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    result = torch.nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)(input_tensor)
    result = result.squeeze()

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
        kernel_size = input_dict["kernel_size"]
        stride = input_dict.get("stride", None)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        ceil_mode = input_dict.get("ceil_mode", False)
        
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        input_tensor = tf.expand_dims(input_tensor, axis=2)

        if stride is None:
            stride = kernel_size
        
        if padding != 0:
            pad_before = pad_after = padding
            input_tensor = tf.pad(input_tensor, [[0, 0], [pad_before, pad_after], [0, 0]], "CONSTANT")

        result = tf.nn.max_pool(
            input_tensor,
            ksize=[1, kernel_size, 1],
            strides=[1, stride, 1],
            padding="VALID" if padding == 0 else "VALID"
        )
        
        if ceil_mode:
            in_len = input_dict["input"].shape[0] + 2 * padding
            out_len = np.floor((in_len - dilation * (kernel_size - 1) - 1) / stride + 1)
            output_size = int(np.ceil((in_len - dilation * (kernel_size - 1) - 1) / stride + 1))
            if output_size > out_len:
                result = result[:, :int(output_size), :, :]

        result = tf.squeeze(result, axis=[0, 2])
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
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
        "return_indices": False,
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
        "return_indices": False,
        "ceil_mode": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()