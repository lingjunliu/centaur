from src.setseed import set_seed
import torch
import tensorflow as tf
import numpy as np

def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    
    # Apply PyTorch's torch.cos
    result = torch.cos(input_tensor)
    
    if not cpu:
        result = result.cpu()
        
    return {"cosine_result": result.numpy()}  # Convert to numpy for comparison

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        
        # Apply TensorFlow's equivalent function
        result = tf.math.cos(input_tensor)
        
        return {"cosine_result": result.numpy()}  # Need to convert to numpy for comparison

def main():
    # Example input
    input_data = {
        "input": np.array([1.4309, 1.2706, -0.8562, 0.9796], dtype=np.float32)
    }

    # PyTorch example
    torch_result = torch_version(input_data)
    print("PyTorch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)
    
    # Assertion - Compare the results
    if np.allclose(torch_result["cosine_result"], tf_result["cosine_result"], atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()