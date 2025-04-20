import torch
import tensorflow as tf
import numpy as np

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["other"])
    
    # Apply torch.fmax
    result_tensor = torch.fmax(input_tensor, other_tensor, out=None)

    if not cpu:
        result_tensor = result_tensor.cpu()

    return {"fmax_result": result_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    device_string = "/cpu:0" if cpu else "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        other_tensor = tf.constant(input["other"])
        
        # Apply tf.maximum
        result_tensor = tf.maximum(input_tensor, other_tensor)

        return {"fmax_result": result_tensor.numpy()}

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "other": np.array([[0.4, 0.5, 0.7], [0.3, 0.4, 1.0]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

if __name__ == "__main__":
    main()
