import numpy as np

def torch_avgpool2d_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input['input'])
    kernel_size = input['kernel_size']
    stride = input.get('stride', kernel_size)
    padding = input.get('padding', 0)
    ceil_mode = input.get('ceil_mode', False)
    count_include_pad = input.get('count_include_pad', True)
    divisor_override = input.get('divisor_override', None)

    # Apply to torch.nn.AvgPool2d
    avg_pool2d = torch.nn.AvgPool2d(kernel_size, stride=stride, padding=padding,
                                    ceil_mode=ceil_mode, count_include_pad=count_include_pad,
                                    divisor_override=divisor_override)
    result = avg_pool2d(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"avg_pool2d_result": result.detach().numpy()}

def tensorflow_avgpool2d_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input['input'])

        # TensorFlow expects channels-last, so transform if necessary
        input_tensor = tf.transpose(input_tensor, perm=[0, 2, 3, 1])

        kernel_size = input['kernel_size']
        stride = input.get('stride', kernel_size)
        padding = input.get('padding', 0)
        ceil_mode = input.get('ceil_mode', False)
        count_include_pad = input.get('count_include_pad', True)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)

        # Calculate padding parameters to match PyTorch behavior
        if padding > 0:
            if ceil_mode:
                padding_type = 'SAME'
            else:
                # Manually pad the input tensor
                input_tensor = tf.pad(input_tensor, [[0, 0], [padding, padding], [padding, padding], [0, 0]], "CONSTANT")
                padding_type = 'VALID'
        else:
            padding_type = 'VALID'

        # Apply to TensorFlow equivalent
        result = tf.nn.avg_pool2d(input_tensor, ksize=kernel_size, strides=stride, padding=padding_type)

        # Transform back to channels-first if necessary
        result = tf.transpose(result, perm=[0, 3, 1, 2])

        return {"avg_pool2d_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(1, 3, 32, 32).astype(np.float32),  # Random 4D tensor
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None
    }

    # Torch example
    torch_result = torch_avgpool2d_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_avgpool2d_version(input_data)
    print("TensorFlow result:", tf_result)

    # Comparing the results
    if np.allclose(torch_result["avg_pool2d_result"], tf_result["avg_pool2d_result"], atol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()