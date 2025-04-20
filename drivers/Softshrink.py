import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    lambd = input.get("lambd", 0.5)

    # Apply PyTorch Softshrink
    softshrink = torch.nn.Softshrink(lambd)
    if not cpu:
        input_tensor = input_tensor.cuda()
        softshrink = softshrink.cuda()

    output = softshrink(input_tensor)

    if not cpu:
        output = output.cpu()

    return {"softshrink_output": output.detach().numpy()}

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
        lambd = input.get("lambd", 0.5)

        # Apply TensorFlow equivalent (Custom implementation for Softshrink)
        output = tf.where(input_tensor > lambd, input_tensor - lambd,
                          tf.where(input_tensor < -lambd, input_tensor + lambd, 0.0))

        return {"softshrink_output": output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[-0.7, 0.3, 0.8], [0.2, -0.6, 0.9]], dtype=np.float32),
        "lambd": 0.5
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Convert both outputs to numpy arrays and compare
    torch_output = torch_result["softshrink_output"]
    tf_output = tf_result["softshrink_output"]

    # Check for equality
    if np.allclose(torch_output, tf_output):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()