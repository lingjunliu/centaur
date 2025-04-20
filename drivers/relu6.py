import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_relu6(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    input_tensor = torch.tensor(input["input"])
    relu6 = torch.nn.ReLU6(inplace=False)

    output_tensor = relu6(input_tensor)

    if cpu:
        output_tensor = output_tensor.cpu()

    return {"torch_relu6": output_tensor.numpy()}

def tensorflow_relu6(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input["input"])
        output_tensor = tf.clip_by_value(input_tensor, clip_value_min=0, clip_value_max=6)

        return {"tensorflow_relu6": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(2, 3).astype(np.float32)
    }

    # Torch example
    torch_result = torch_relu6(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_relu6(input_data)
    print("TensorFlow result:", tf_result)

    # Assert and compare snapshots
    assert np.allclose(torch_result["torch_relu6"], tf_result["tensorflow_relu6"]), "Results are not equal!"
    print("Results are equal")

if __name__ == "__main__":
    main()