import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["other"])

    # Apply to torch.logaddexp
    if cpu:
        result = torch.logaddexp(input_tensor, other_tensor).cpu().numpy()
    else:
        result = torch.logaddexp(input_tensor.cuda(), other_tensor.cuda()).cpu().numpy()
    
    return {"logaddexp_result": result}

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
        other_tensor = tf.constant(input["other"])

        # Apply to TensorFlow equivalent
        logaddexp_result = tf.math.reduce_logsumexp(
            tf.stack([input_tensor, other_tensor], axis=-1), axis=-1
        )

        result = logaddexp_result.numpy()
        
    return {"logaddexp_result": result}

def main():
    # Example input
    input_data = {
        "input": np.array([-1.0, -200.0, 30000.0], dtype=np.float32),
        "other": np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert equality
    torch_result_array = np.array(torch_result["logaddexp_result"])
    tf_result_array = np.array(tf_result["logaddexp_result"])

    if np.allclose(torch_result_array, tf_result_array):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()