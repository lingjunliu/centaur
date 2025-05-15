import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", None)
    padding = input_dict.get("padding", 0)
    ceil_mode = input_dict.get("ceil_mode", False)
    count_include_pad = input_dict.get("count_include_pad", True)
    divisor_override = input_dict.get("divisor_override", None)

    if not cpu:
        input_tensor = input_tensor.cuda()

    avg_pool = torch.nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override)
    result = avg_pool(input_tensor)

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
        ceil_mode = input_dict.get("ceil_mode", False)
        count_include_pad = input_dict.get("count_include_pad", True)
        divisor_override = input_dict.get("divisor_override", None)

        if isinstance(kernel_size, int):
            kernel_size_tuple = (kernel_size, kernel_size, kernel_size)
        else:
            kernel_size_tuple = tuple(kernel_size)

        if stride is None:
            stride_tuple = kernel_size_tuple
        elif isinstance(stride, int):
            stride_tuple = (stride, stride, stride)
        else:
            stride_tuple = tuple(stride)

        if isinstance(padding, int):
            padding_tuple = (padding, padding, padding)
        else:
            padding_tuple = tuple(padding)

        input_shape = input_tensor.shape
        input_tensor = tf.pad(input_tensor, [[0, 0], [0, 0], [padding_tuple[0], padding_tuple[0]], [padding_tuple[1], padding_tuple[1]], [padding_tuple[2], padding_tuple[2]]])
        
        def calculate_output_shape(input_size, kernel_size, stride, padding, ceil_mode):
            output_size = (input_size + 2 * padding - kernel_size) / stride + 1
            if ceil_mode:
                return int(np.ceil(output_size))
            else:
                return int(np.floor(output_size))

        d_out = calculate_output_shape(input_shape[2], kernel_size_tuple[0], stride_tuple[0], padding_tuple[0], ceil_mode)
        h_out = calculate_output_shape(input_shape[3], kernel_size_tuple[1], stride_tuple[1], padding_tuple[1], ceil_mode)
        w_out = calculate_output_shape(input_shape[4], kernel_size_tuple[2], stride_tuple[2], padding_tuple[2], ceil_mode)

        output = np.zeros((input_shape[0], input_shape[1], d_out, h_out, w_out))

        for n in range(input_shape[0]):
            for c in range(input_shape[1]):
                for d in range(d_out):
                    for h in range(h_out):
                        for w in range(w_out):
                            d_start = d * stride_tuple[0]
                            d_end = d_start + kernel_size_tuple[0]
                            h_start = h * stride_tuple[1]
                            h_end = h_start + kernel_size_tuple[1]
                            w_start = w * stride_tuple[2]
                            w_end = w_start + kernel_size_tuple[2]

                            window = input_tensor[n, c, d_start:d_end, h_start:h_end, w_start:w_end].numpy()

                            if count_include_pad:
                                divisor = kernel_size_tuple[0] * kernel_size_tuple[1] * kernel_size_tuple[2]
                            else:
                                valid_elements = np.sum(window != 0)
                                divisor = valid_elements if valid_elements > 0 else 1
                            
                            if divisor_override is not None:
                                divisor = divisor_override

                            output[n, c, d, h, w] = np.sum(window) / divisor
        
        result = output

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 10, 10, 10).astype(np.float32),
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(2, 3, 10, 10, 10).astype(np.float32),
        "kernel_size": (3, 2, 2),
        "stride": (2, 1, 2),
        "padding": (1, 0, 1),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()