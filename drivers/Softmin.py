import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    input_tensor = torch.tensor(input["input"])
    dim = input.get("dim", None)

    if dim is None:
        m = torch.nn.Softmin()
    else:
        # Apply Softmin
        m = torch.nn.Softmin(dim=dim)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        m = m.cuda()
    
    output = m(input_tensor)

    if not cpu:
        output = output.cpu()

    return {"softmin_output": output.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input["input"])
        dim = input.get("dim", None)

        if dim is None:
            dim = -1  # Default dimension to apply Softmin

        # Calculate Softmin
        exp_neg_x = tf.exp(-input_tensor)
        sum_exp_neg_x = tf.reduce_sum(exp_neg_x, axis=dim, keepdims=True)
        output = exp_neg_x / sum_exp_neg_x

        return {"softmin_output": output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(2, 3).astype(np.float32),
        "dim": 1  # Apply softmin along this dimension
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_output = torch_result["softmin_output"]
    tf_output = tf_result["softmin_output"]

    if np.allclose(torch_output, tf_output, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()