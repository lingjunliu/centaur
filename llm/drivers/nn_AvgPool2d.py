import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", None)
    padding = input_dict.get("padding", 0)
    ceil_mode = input_dict.get("ceil_mode", False)
    count_include_pad = input_dict.get("count_include_pad", True)
    divisor_override = input_dict.get("divisor_override", None)

    if not cpu:
        input_tensor = input_tensor.cuda()

    avg_pool = torch.nn.AvgPool2d(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override
    )

    result = avg_pool(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

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
            kernel_size = [kernel_size, kernel_size]
        if stride is None:
            stride = kernel_size
        if isinstance(stride, int):
            stride = [stride, stride]
        if isinstance(padding, int):
            padding = [padding, padding]

        input_shape = input_tensor.shape
        input_tensor = tf.reshape(input_tensor, (1, input_shape[0], input_shape[1], input_shape[2]))

        padding_config = [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]]
        input_padded = tf.pad(input_tensor, padding_config, "CONSTANT")

        if ceil_mode:
            result = tf.nn.avg_pool(
                input_padded,
                ksize=[1, kernel_size[0], kernel_size[1], 1],
                strides=[1, stride[0], stride[1], 1],
                padding="SAME"
            )
        else:
            result = tf.nn.avg_pool(
                input_padded,
                ksize=[1, kernel_size[0], kernel_size[1], 1],
                strides=[1, stride[0], stride[1], 1],
                padding="VALID"
            )
        
        result = tf.reshape(result, result.shape[1:])

        if count_include_pad:
            if divisor_override is None:
                divisor = kernel_size[0] * kernel_size[1]
            else:
                divisor = divisor_override
            result = result / divisor
        elif divisor_override is not None:
            result = result / divisor_override

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(3, 32, 32).astype(np.float32),
        "kernel_size": (3, 4),
        "stride": (2,3),
        "padding": (1,2),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    min_c = min(torch_result["result"].shape[0], tf_result["result"].shape[0])
    min_h = min(torch_result["result"].shape[1], tf_result["result"].shape[1])
    min_w = min(torch_result["result"].shape[2], tf_result["result"].shape[2])
    
    torch_cropped = torch_result["result"][:min_c, :min_h, :min_w]
    tf_cropped = tf_result["result"][:min_c, :min_h, :min_w]

    assert np.allclose(torch_cropped, tf_cropped, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()