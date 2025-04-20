import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_min_scalar_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    # Apply torch.min
    min_value = torch.min(input_tensor)

    if not cpu:
        min_value = min_value.cpu()

    return {"min_value": float(min_value.item())}

def tensorflow_min_scalar_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # Apply tf.reduce_min
        min_value = tf.reduce_min(input_tensor)

        return {"min_value": float(min_value.numpy())}

def torch_min_dimensional_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    dim = input.get("dim", 0)
    keepdim = input.get("keepdim", False)

    # Apply torch.min with dim
    min_values, min_indices = torch.min(input_tensor, dim=dim, keepdim=keepdim)

    if not cpu:
        min_values = min_values.cpu()
        min_indices = min_indices.cpu()

    return {
        "min_values": min_values.numpy().tolist(),
        "min_indices": min_indices.numpy().tolist(),
    }

def tensorflow_min_dimensional_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        dim = input.get("dim", 0)
        keepdim = input.get("keepdim", False)

        # Apply tf.reduce_min with axis
        min_values = tf.reduce_min(input_tensor, axis=dim, keepdims=keepdim)
        min_indices = tf.argmin(input_tensor, axis=dim)

        return {
            "min_values": min_values.numpy().tolist(),
            "min_indices": min_indices.numpy().tolist(),
        }

def torch_min_elementwise_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor1 = torch.tensor(input["input1"])
    input_tensor2 = torch.tensor(input["input2"])

    # Apply torch.min for element-wise comparison
    min_values = torch.min(input_tensor1, input_tensor2)

    if not cpu:
        min_values = min_values.cpu()

    return {"min_values": min_values.numpy().tolist()}

def tensorflow_min_elementwise_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor1 = tf.constant(input["input1"])
        input_tensor2 = tf.constant(input["input2"])

        # Apply tf.math.minimum for element-wise comparison
        min_values = tf.math.minimum(input_tensor1, input_tensor2)

        return {"min_values": min_values.numpy().tolist()}

def main():
    # Example input for scalar min
    input_data_scalar = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32)
    }

    torch_result_scalar = torch_min_scalar_version(input_data_scalar)
    tf_result_scalar = tensorflow_min_scalar_version(input_data_scalar)

    # Example input for dimensional min
    input_data_dimensional = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "dim": 1,
        "keepdim": False
    }

    torch_result_dimensional = torch_min_dimensional_version(input_data_dimensional)
    tf_result_dimensional = tensorflow_min_dimensional_version(input_data_dimensional)

    # Example input for element-wise min
    input_data_elementwise = {
        "input1": np.array([[0.5, 0.3, 0.8], [0.7, 0.6, 0.4]], dtype=np.float32),
        "input2": np.array([[0.2, 0.4, 0.9], [0.1, 0.5, 0.6]], dtype=np.float32)
    }

    torch_result_elementwise = torch_min_elementwise_version(input_data_elementwise)
    tf_result_elementwise = tensorflow_min_elementwise_version(input_data_elementwise)

    # Compare results
    assert np.isclose(torch_result_scalar["min_value"], tf_result_scalar["min_value"]), "Scalar min results do not match"
    print("Scalar min: equal")

    assert np.allclose(torch_result_dimensional["min_values"], tf_result_dimensional["min_values"]), "Dimensional min values do not match"
    assert np.array_equal(torch_result_dimensional["min_indices"], tf_result_dimensional["min_indices"]), "Dimensional min indices do not match"
    print("Dimensional min: equal")

    assert np.allclose(torch_result_elementwise["min_values"], tf_result_elementwise["min_values"]), "Element-wise min values do not match"
    print("Element-wise min: equal")

if __name__ == "__main__":
    main()