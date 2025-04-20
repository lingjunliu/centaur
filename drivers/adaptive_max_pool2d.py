import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    
    # Ensure the tensor is on the right device
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply PyTorch adaptive max pooling
    output_size = input["output_size"]
    
    result = torch.nn.functional.adaptive_max_pool2d(input_tensor, output_size)
    
    if not cpu:
        result = result.cpu()

    return {"adaptive_max_pool2d_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        
        # Ensure the tensor is formatted correctly
        input_tensor = tf.transpose(input_tensor, [0, 2, 3, 1])  # Convert from Channel First to Channel Last for TensorFlow
        
        # Apply TensorFlow equivalent of adaptive max pooling
        output_size = input["output_size"]
        result = tf.nn.max_pool(input_tensor, ksize=[1, input_tensor.shape[1] // output_size[0], input_tensor.shape[2] // output_size[1], 1],
                                strides=[1, input_tensor.shape[1] // output_size[0], input_tensor.shape[2] // output_size[1], 1], padding='VALID')
        
        result = tf.transpose(result, [0, 3, 1, 2])  # Convert back to Channel First format
        
        return {"adaptive_max_pool2d_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),  # Example 3-channel 32x32 input image
        "output_size": (16, 16)  # Example output size
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Use numpy for comparison
    are_close = np.allclose(
        torch_result["adaptive_max_pool2d_result"],
        tf_result["adaptive_max_pool2d_result"],
        atol=1e-5
    )
    
    if are_close:
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()