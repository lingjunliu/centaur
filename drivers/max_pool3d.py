import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    kernel_size = input.get("kernel_size")
    stride = input.get("stride", [1, 1, 1])
    padding = input.get("padding", 0)
    dilation = input.get("dilation", 1)
    ceil_mode = input.get("ceil_mode", False)
    
    # PyTorch expects input in the format (N, C, D, H, W)
    input_tensor = input_tensor.permute(0, 4, 1, 2, 3)

    # Apply torch max_pool3d
    if not cpu:
        input_tensor = input_tensor.cuda()
    result = torch.nn.functional.max_pool3d(
        input_tensor, kernel_size, stride=stride, padding=padding,
        dilation=dilation, ceil_mode=ceil_mode
    )

    if not cpu:
        result = result.cpu()

    # Permute back to original format (N, D, H, W, C)
    result = result.permute(0, 2, 3, 4, 1)

    return {"max_pool3d_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        ksize = [1] + input.get("kernel_size", [1, 1, 1]) + [1]
        strides = [1] + input.get("stride", [1, 1, 1]) + [1]
        padding = 'SAME' if input.get("padding", 0) > 0 else 'VALID'
        data_format = 'NDHWC'  # TensorFlow expects input in the format (N, D, H, W, C)

        # Apply TensorFlow max_pool3d
        result = tf.nn.max_pool3d(
            input=input_tensor,
            ksize=ksize,
            strides=strides,
            padding=padding,
            data_format=data_format
        )

        return {"max_pool3d_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.rand(2, 4, 4, 4, 3).astype(np.float32),  # Shape: (batch, depth, height, width, channels)
        "kernel_size": [2, 2, 2],
        "stride": [2, 2, 2],
        "padding": 0,
        "dilation": 1,
        "ceil_mode": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert equal results
    if np.allclose(torch_result["max_pool3d_result"], tf_result["max_pool3d_result"], atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()