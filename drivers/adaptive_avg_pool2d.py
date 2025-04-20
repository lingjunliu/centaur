import numpy as np
import random

# Mock set_seed function for reproducibility if src.setseed is not available
def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply to torch.nn.functional.adaptive_avg_pool2d
    output_size = input["output_size"]
    result = torch.nn.functional.adaptive_avg_pool2d(input_tensor, output_size)

    if not cpu:
        result = result.cpu()

    return {"adaptive_avg_pool2d_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        output_size = input["output_size"]

        # Permute dimensions from NCHW to NHWC
        input_tensor = tf.transpose(input_tensor, perm=[0, 2, 3, 1])

        # Calculate kernel sizes and strides based on the desired output size
        input_size = input_tensor.shape[1:3]  # Assuming NHWC format
        stride = [
            input_size[0] // output_size[0],
            input_size[1] // output_size[1]
        ]
        kernel_size = [
            input_size[0] - (output_size[0] - 1) * stride[0],
            input_size[1] - (output_size[1] - 1) * stride[1]
        ]

        # Apply to TensorFlow equivalent using tf.nn.avg_pool
        result = tf.nn.avg_pool(
            input_tensor,  # NHWC
            ksize=[1, kernel_size[0], kernel_size[1], 1],
            strides=[1, stride[0], stride[1], 1],
            padding='VALID'
        )

        # Permute dimensions back to NCHW
        result = tf.transpose(result, perm=[0, 3, 1, 2])

        return {"adaptive_avg_pool2d_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.rand(1, 1, 32, 32).astype(np.float32),  # NCHW format
        "output_size": (16, 16)
    }

    # Torch example
    torch_result = torch_version(input_data, cpu=True)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data, cpu=True)
    print("TensorFlow result:", tf_result)

    # Assertion to compare results
    np.testing.assert_allclose(
        torch_result["adaptive_avg_pool2d_result"],
        tf_result["adaptive_avg_pool2d_result"],
        rtol=1e-5,
        atol=1e-5
    )

    print("Results are equal")

if __name__ == "__main__":
    main()