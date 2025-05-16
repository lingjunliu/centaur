import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    
    kernel_size = input.get("kernel_size", 2)
    stride = input.get("stride", 2)
    padding = input.get("padding", 0)
    dilation = input.get("dilation", 1)
    ceil_mode = input.get("ceil_mode", False)
    return_indices = input.get("return_indices", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply to torch.nn.functional.max_pool2d
    output = torch.nn.functional.max_pool2d(
        input_tensor, kernel_size, stride, padding, 
        dilation, ceil_mode, return_indices
    )

    if not cpu:
        output = output.cpu()

    if return_indices:
        return {"output": output[0].detach().numpy().tolist(), "indices": output[1].detach().numpy().tolist()}
    else:
        return {"output": output.detach().numpy().tolist()}

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
        
        pool_size = input.get("kernel_size", 2)
        strides = input.get("stride", 2)
        padding = 'VALID' if input.get("padding", 0) == 0 else 'SAME'
        
        # Transpose for TensorFlow compatibility (NCHW to NHWC)
        input_tensor = tf.transpose(input_tensor, perm=[0, 2, 3, 1])

        output = tf.nn.max_pool2d(
            input_tensor, ksize=pool_size, strides=strides, padding=padding
        )
        
        # Transpose back to original format (NHWC to NCHW)
        output = tf.transpose(output, perm=[0, 3, 1, 2])
        
        return {"output": output.numpy().tolist()}

def main():
    # Example input
    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "kernel_size": 2,
        "stride": 2,
        "padding": 0, 
        "dilation": 1,
        "ceil_mode": False,
        "return_indices": False
    }

    # Torch example
    torch_result = torch_version(input_data, cpu=True)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data, cpu=True)
    print("TensorFlow result:", tf_result)

    # Compare the results
    # np.testing.assert_almost_equal(
    #     np.array(torch_result["output"]),
    #     np.array(tf_result["output"]),
    #     decimal=5
    # )

    # print("equal")
    if np.allclose(np.array(torch_result["output"]), np.array(tf_result["output"]), atol=1e-5):
        print("equal")
    else:
        print("not equal")


if __name__ == "__main__":
    main()