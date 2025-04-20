import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()
    
    # Extract parameters from input dictionary
    input_tensor = torch.tensor(input["input"])
    k = input["k"]
    dim = input.get("dim", None)
    largest = input.get("largest", True)
    sorted = input.get("sorted", True)
    
    # Apply torch.topk
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    if dim is not None:
        values, indices = torch.topk(input_tensor, k, dim=dim, largest=largest, sorted=sorted)
    else:
        values, indices = torch.topk(input_tensor, k, largest=largest, sorted=sorted)

    if not cpu:
        values = values.cpu()
        indices = indices.cpu()

    return {"values": values.numpy(), "indices": indices.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    # Extract parameters from input dictionary
    input_tensor = tf.constant(input["input"])
    k = input["k"]
    dim = input.get("dim", -1)
    largest = input.get("largest", True)
    sorted = input.get("sorted", True)
    
    with tf.device(device_string):
        if largest:
            values, indices = tf.math.top_k(input_tensor, k=k, sorted=sorted)
        else:
            neg_input_tensor = -input_tensor
            neg_values, indices = tf.math.top_k(neg_input_tensor, k=k, sorted=sorted)
            values = -neg_values
    
    return {"values": values.numpy(), "indices": indices.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "k": 2,
        "dim": 1,
        "largest": True,
        "sorted": True
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch values:", torch_result["values"])
    print("Torch indices:", torch_result["indices"])

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow values:", tf_result["values"])
    print("TensorFlow indices:", tf_result["indices"])

    # Check equality
    values_equal = np.allclose(torch_result["values"], tf_result["values"])
    indices_equal = np.array_equal(torch_result["indices"], tf_result["indices"])

    if values_equal and indices_equal:
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()