import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    dim = input.get("dim", None)
    
    if dim is not None:
        softmax = torch.nn.Softmax(dim=dim)
    else:
        softmax = torch.nn.Softmax()

    if not cpu:
        input_tensor = input_tensor.cuda()
        softmax = softmax.cuda()

    # Apply Softmax
    result = softmax(input_tensor)

    if not cpu:
        result = result.cpu()
    
    return {"softmax": result.numpy()}

def tensorflow_version(input, cpu=True):
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        dim = input.get("dim", None)
        
        if dim is not None:
            result = tf.nn.softmax(input_tensor, axis=dim)
        else:
            raise ValueError("The dimension 'dim' must be provided for TensorFlow Softmax")
        
        return {"softmax": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "dim": 1
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Comparison
    if np.allclose(torch_result["softmax"], tf_result["softmax"], atol=1e-7):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()