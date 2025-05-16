import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    norm_type = input["norm_type"]
    kernel_size = input["kernel_size"]
    stride = input.get("stride", None)
    ceil_mode = input.get("ceil_mode", False)
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply to torch.nn.functional.lp_pool2d
    pooled = torch.nn.functional.lp_pool2d(
        input_tensor, norm_type, kernel_size, stride=stride, ceil_mode=ceil_mode
    )

    if not cpu:
        pooled = pooled.cpu()

    return {"lp_pool2d_result": pooled.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        norm_type = input["norm_type"]
        kernel_size = input["kernel_size"]
        stride = input.get("stride", None) or kernel_size  # Default to kernel_size if stride is None
        ceil_mode = input.get("ceil_mode", False)

        # TensorFlow does not have a direct equivalent to lp_pool2d, so we simulate it using conv2d and complex operations:
        # To simulate the Lp pooling (p-norm pooling), we can use TensorFlow operations

        # Raise input tensor elements to the power of norm_type
        powered_input = tf.abs(input_tensor) ** norm_type
        
        # Apply average pooling (identity for norm_type=2)
        pooled = tf.nn.avg_pool2d(powered_input, ksize=kernel_size, strides=stride, padding="SAME" if ceil_mode else "VALID")
        
        # Take the p-th root of the pooled results
        pooled = pooled ** (1.0 / norm_type)

        return {"lp_pool2d_result": pooled.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=np.float32),  # Shape (1, 1, 3, 3)
        "norm_type": 2.0,
        "kernel_size": 2,
        "stride": None,
        "ceil_mode": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_array = torch_result["lp_pool2d_result"]
    tf_array = tf_result["lp_pool2d_result"]

    if np.allclose(torch_array, tf_array, rtol=1e-05, atol=1e-08):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()