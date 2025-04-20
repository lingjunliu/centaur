import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    output_size = input["output_size"]
    
    # Move to CUDA if needed
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply to torch.nn.functional.adaptive_avg_pool1d
    result = torch.nn.functional.adaptive_avg_pool1d(input_tensor, output_size)

    if not cpu:
        result = result.cpu()

    return {"adaptive_avg_pool1d": result.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Select device based on `cpu` flag
    device_string = "/cpu:0" if cpu else "/gpu:0"
    
    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        output_size = input["output_size"]

        input_length = input_tensor.shape[2]
        stride = input_length // output_size
        kernel_size = input_length - (output_size - 1) * stride
        
        input_tensor = tf.transpose(input_tensor, perm=[0, 2, 1]) # Transpose to (batch, length, channels)
        
        pooled = tf.nn.pool(input_tensor, window_shape=[kernel_size], pooling_type='AVG', strides=[stride], padding='VALID')
        
        # Transpose back to (batch, channels, length)
        pooled = tf.transpose(pooled, perm=[0, 2, 1])
        
    return {"adaptive_avg_pool1d": pooled.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.rand(1, 3, 8).astype(np.float32),  # (batch_size, channels, length)
        "output_size": 4
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert equality within a tolerance
    is_equal = np.allclose(torch_result["adaptive_avg_pool1d"], tf_result["adaptive_avg_pool1d"], atol=1e-5)
    print("equal" if is_equal else "not equal")


if __name__ == "__main__":
    main()