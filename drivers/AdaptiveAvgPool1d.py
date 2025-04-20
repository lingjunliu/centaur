import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_adaptive_avg_pool1d(input, cpu=True):
    # Set seed for reproducibility
    set_seed()
    
    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    output_size = input["output_size"]

    # Move tensor to appropriate device
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply AdaptiveAvgPool1d
    pool = torch.nn.AdaptiveAvgPool1d(output_size)
    output = pool(input_tensor)

    if not cpu:
        output = output.cpu()

    return {"output": output.detach().numpy()}

def tensorflow_adaptive_avg_pool1d(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        output_size = input["output_size"]

        # Extract input dimensions
        input_shape = tf.shape(input_tensor)
        batch_size, channels, length = input_shape[0], input_shape[1], input_shape[2]

        # Calculate pool_size and strides to match the output size
        pool_size = [length // output_size] if length // output_size > 0 else [1]
        strides = [length // output_size] if length // output_size > 0 else [1]

        # Use AveragePooling1D layer with dynamic pool_size and strides
        pool = tf.keras.layers.AveragePooling1D(pool_size=pool_size, strides=strides, padding='valid')
        output = pool(tf.transpose(input_tensor, [0, 2, 1]))  # Transpose to (batch_size, length, channels to apply pooling)
        
        # Calculate the exact number of output elements to match
        exact_output_size = tf.cast(tf.math.ceil(length / pool_size[0]), tf.int32)
        if exact_output_size > output_size:
            output = output[:, :output_size, :]
        
        output = tf.transpose(output, [0, 2, 1])  # Transpose back to (batch_size, channels, output_size)

        return {"output": output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.rand(1, 64, 8).astype(np.float32),  # Replace with a specific random seed if needed
        "output_size": 4  # Adjusted the output size to effectively showcase matching
    }

    # Torch example
    torch_result = torch_adaptive_avg_pool1d(input_data)
    print("Torch result:", torch_result['output'])

    # TensorFlow example
    tf_result = tensorflow_adaptive_avg_pool1d(input_data)
    print("TensorFlow result:", tf_result['output'])

    # Compare the results
    if np.allclose(torch_result["output"], tf_result["output"], atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()