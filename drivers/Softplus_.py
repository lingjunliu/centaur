import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    beta = input.get("beta", 1.0)
    threshold = input.get("threshold", 20.0)

    # Softplus function
    softplus_function = torch.nn.Softplus(beta=beta, threshold=threshold)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        softplus_function = softplus_function.cuda()
    
    result = softplus_function(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"Softplus_output": result.numpy()}

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
        beta = input.get("beta", 1.0)
        threshold = input.get("threshold", 20.0)

        def softplus(x, beta, threshold):
            beta_x = beta * x
            return tf.where(beta_x > threshold, x, tf.math.log(1 + tf.math.exp(beta_x)) / beta)

        result = softplus(input_tensor, beta=beta, threshold=threshold)

        return {"Softplus_output": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 3.0]], dtype=np.float32),
        "beta": 1.0,
        "threshold": 20.0
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result["Softplus_output"])

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result["Softplus_output"])

    if np.allclose(torch_result["Softplus_output"], tf_result["Softplus_output"], atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()