import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply torch.sigmoid function
    loss = torch.sigmoid(input_tensor)

    if not cpu:
        loss = loss.cpu()

    return {"sigmoid_result": loss.numpy()}

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

        # TensorFlow does not have an out parameter for sigmoid, so we just compute
        loss = tf.sigmoid(input_tensor)

        return {"sigmoid_result": loss.numpy()}

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

    if np.allclose(torch_result["sigmoid_result"], tf_result["sigmoid_result"], rtol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()