from src.setseed import set_seed
import torch
import tensorflow as tf
import numpy as np

# Dummy set_seed function for reproducibility. Implement as needed.
def set_seed(seed=42):
    torch.manual_seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    # Apply torch.prod based on provided parameters
    if 'dim' in input:
        prod_result = torch.prod(input_tensor, dim=input["dim"], keepdim=input.get("keepdim", False))
    else:
        prod_result = torch.prod(input_tensor)

    if not cpu:
        prod_result = prod_result.cpu()

    return {"prod_result": prod_result.numpy()}

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

        # Apply TensorFlow equivalent based on provided parameters
        if 'dim' in input:
            prod_result = tf.reduce_prod(input_tensor, axis=input["dim"], keepdims=input.get("keepdim", False))
        else:
            prod_result = tf.reduce_prod(input_tensor)

        return {"prod_result": prod_result.numpy()}

def main():
    # Example 1: Product of all elements
    input_data_1 = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32)
    }

    # Example 2: Product across a specific dimension
    input_data_2 = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "dim": 1,
        "keepdim": False
    }

    # Torch Example 1
    torch_result_1 = torch_version(input_data_1)
    print("Torch result 1:", torch_result_1)

    # TensorFlow Example 1
    tf_result_1 = tensorflow_version(input_data_1)
    print("TensorFlow result 1:", tf_result_1)

    # Assert Example 1
    assert np.allclose(torch_result_1["prod_result"], tf_result_1["prod_result"]), "Example 1: Results do not match"
    print("Example 1: equal")

    # Torch Example 2
    torch_result_2 = torch_version(input_data_2)
    print("Torch result 2:", torch_result_2)

    # TensorFlow Example 2
    tf_result_2 = tensorflow_version(input_data_2)
    print("TensorFlow result 2:", tf_result_2)

    # Assert Example 2
    assert np.allclose(torch_result_2["prod_result"], tf_result_2["prod_result"]), "Example 2: Results do not match"
    print("Example 2: equal")

if __name__ == "__main__":
    main()