import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    lower = input.get("lower", 1./8)
    upper = input.get("upper", 1./3)
    training = input.get("training", False)
    inplace = input.get("inplace", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply to torch.nn.functional.rrelu
    result = torch.nn.functional.rrelu(
        input_tensor, lower=lower, upper=upper, training=training, inplace=inplace
    )

    if not cpu:
        result = result.cpu()

    return {"rrelu_result": result.numpy()}

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
        alpha = (input.get("lower", 1./8) + input.get("upper", 1./3)) / 2  # Using an average value

        # Apply to TensorFlow equivalent
        result = tf.nn.leaky_relu(input_tensor, alpha=alpha)

        return {"rrelu_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "lower": 1./8,
        "upper": 1./3,
        "training": True,
        "inplace": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    assert np.allclose(torch_result["rrelu_result"], tf_result["rrelu_result"], atol=1e-5), "not equal"
    print("equal")

if __name__ == "__main__":
    main()