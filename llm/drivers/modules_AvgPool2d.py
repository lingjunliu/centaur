import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", kernel_size)
    padding = input_dict.get("padding", 0)
    ceil_mode = input_dict.get("ceil_mode", False)
    count_include_pad = input_dict.get("count_include_pad", True)
    divisor_override = input_dict.get("divisor_override", None)

    if not cpu:
        input_tensor = input_tensor.cuda()

    avg_pool = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override)
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
        stride = input_dict.get("stride", kernel_size)
        padding = input_dict.get("padding", 0)
        ceil_mode = input_dict.get("ceil_mode", False)
        count_include_pad = input_dict.get("count_include_pad", True)
        divisor_override = input_dict.get("divisor_override", None)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        if isinstance(stride, int):
            stride = [stride, stride]
        if isinstance(padding, int):
            padding = [padding, padding]

        input_tensor = tf.cast(input_tensor, tf.float32)
        input_tensor = tf.transpose(input_tensor, [0, 2, 3, 1])

        if padding != [0, 0]:
            input_tensor = tf.pad(input_tensor, [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]])

        result = tf.nn.avg_pool(
            input_tensor,
            ksize=[1, kernel_size[0], kernel_size[1], 1],
            strides=[1, stride[0], stride[1], 1],
            padding='VALID'
        )

        result = tf.transpose(result, [0, 3, 1, 2])
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
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

    print("Success")

if __name__ == "__main__":
    main()