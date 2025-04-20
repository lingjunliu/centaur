import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    inplace = input.get("inplace", False)

    # Apply torch.nn.Hardswish
    activation = torch.nn.Hardswish(inplace=inplace)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        activation = activation.cuda()

    result = activation(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"hardswish": result.detach().numpy()}

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

        # TensorFlow equivalent of Hardswish
        def tf_hardswish(x):
            return x * tf.nn.relu6(x + 3) / 6

        result = tf_hardswish(input_tensor)

        return {"hardswish": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, -0.3, 3.8], [-2.2, 2.6, 0.9]], dtype=np.float32),
        "inplace": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    if np.allclose(torch_result["hardswish"], tf_result["hardswish"], atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()