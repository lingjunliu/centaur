import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_adaptive_max_pool1d(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    output_size = input["output_size"]

    # Apply to torch.nn.functional.adaptive_max_pool1d
    pooled_output = torch.nn.functional.adaptive_max_pool1d(input_tensor, output_size)

    if not cpu:
        pooled_output = pooled_output.cpu()

    return {"adaptive_max_pool1d_output": pooled_output.numpy()}

def tensorflow_adaptive_max_pool1d(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.convert_to_tensor(input["input"])
        output_size = input["output_size"][0]  # TensorFlow pooling size needs a single integer
        
        input_shape = tf.shape(input_tensor)
        batch_size, channels, input_length = input_shape[0], input_shape[1], input_shape[2]
        
        # Calculate kernel size and strides to roughly match adaptive pooling
        kernel_size = input_length // output_size
        stride_size = input_length // output_size
        
        # Apply multiple max pooling operations to simulate adaptive pooling
        pooled_output = tf.reshape(input_tensor, [batch_size, channels, input_length // kernel_size, kernel_size])
        pooled_output = tf.reduce_max(pooled_output, axis=-1) 
        
        return {"adaptive_max_pool1d_output": pooled_output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.rand(1, 3, 12).astype(np.float32),  # Example input tensor
        "output_size": (4,)  # Desired output size from pooling
    }

    # Torch example
    torch_result = torch_adaptive_max_pool1d(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_adaptive_max_pool1d(input_data)
    print("TensorFlow result:", tf_result)

    # Assuming we have numpy for comparison
    torch_output = torch_result["adaptive_max_pool1d_output"]
    tf_output = tf_result["adaptive_max_pool1d_output"]

    if np.allclose(torch_output, tf_output, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()