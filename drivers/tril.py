from src.setseed import set_seed
import torch
import tensorflow as tf
import numpy as np

# Assume set_seed is defined in src.setseed
def set_seed(seed=1234):
    torch.manual_seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    input_tensor = torch.tensor(input['input'])
    diagonal = input.get('diagonal', 0)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    # Apply torch.tril
    output_tensor = torch.tril(input_tensor, diagonal=diagonal)
    
    if not cpu:
        output_tensor = output_tensor.cpu()
        
    return {"tril_output": output_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input['input'])
        diagonal = input.get('diagonal', 0)
        
        # Apply tf.experimental.numpy.tril
        # TensorFlow does not have a direct tril equivalent, we need to use numpy functions
        output_tensor = tf.experimental.numpy.tril(input_tensor, k=diagonal)
    
    return {"tril_output": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(4, 6).astype(np.float32),
        "diagonal": 1
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)
    
    # Compare the results
    torch_output = np.array(torch_result["tril_output"])
    tf_output = np.array(tf_result["tril_output"])
    
    if np.allclose(torch_output, tf_output):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()