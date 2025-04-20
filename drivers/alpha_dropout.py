import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    p = input.get("p", 0.5)
    training = input.get("training", False)
    inplace = input.get("inplace", False)

    # Apply to torch.nn.functional.alpha_dropout
    result = torch.nn.functional.alpha_dropout(
        input_tensor, p=p, training=training, inplace=inplace
    )

    if not cpu:
        result = result.cpu()

    return {"alpha_dropout_result": result.numpy()}

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
        p = input.get("p", 0.5)
        training = input.get("training", False)
        inplace = input.get("inplace", False)
        
        # Apply equivalent to TensorFlow: tf.nn.alpha_dropout
        # TensorFlow does not have an exact equivalent for alpha_dropout, but we can use dropout with scale factor
        # for comparison purposes when training is False it should return the input as it is.
        if training:
            result = tf.nn.dropout(input_tensor, rate=p) * tf.math.sqrt(1.0 / (1.0 - p))
        else:
            result = input_tensor

        return {"alpha_dropout_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "p": 0.5,
        "training": False,
        "inplace": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results and print 'equal' or 'not equal'
    if np.allclose(torch_result["alpha_dropout_result"], tf_result["alpha_dropout_result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()