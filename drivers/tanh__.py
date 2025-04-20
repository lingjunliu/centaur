import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()
    
    # Ensure the operation runs on CPU if specified
    if cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
    # Unpack input dictionary and create tensor
    input_tensor = torch.tensor(input["input"])
    input_tensor = input_tensor.to(device)
    
    # Apply the Tanh function
    output_tensor = torch.nn.functional.tanh(input_tensor)
    output_tensor = output_tensor.cpu()  # Ensure the output is on the CPU
    
    return {"tanh_output": output_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    device_string = "/cpu:0" if cpu else "/gpu:0"
    
    with tf.device(device_string):
        # Unpack input dictionary and create tensor
        input_tensor = tf.constant(input["input"])
        
        # Apply the Tanh function
        output_tensor = tf.nn.tanh(input_tensor)
        
        # Note: In TensorFlow, there's no need to manually transfer to CPU, but we will get the numpy array for comparison
        return {"tanh_output": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results using allclose because floating point operations might have minor differences
    if np.allclose(torch_result["tanh_output"], tf_result["tanh_output"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()
