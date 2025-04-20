import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    dim = input.get("dim", -1)
    descending = input.get("descending", False)
    stable = input.get("stable", False)

    # Apply torch.sort
    sorted_tensor, indices = torch.sort(input_tensor, dim=dim, descending=descending, stable=stable)

    if not cpu:
        sorted_tensor = sorted_tensor.cpu()
        indices = indices.cpu()

    return {
        "sorted_values": sorted_tensor.numpy(),
        "indices": indices.numpy()
    }

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        dim = input.get("dim", -1)
        descending = input.get("descending", False)

        # Apply tf.sort and tf.argsort
        sorted_tensor = tf.sort(input_tensor, axis=dim, direction='DESCENDING' if descending else 'ASCENDING')
        indices = tf.argsort(input_tensor, axis=dim, direction='DESCENDING' if descending else 'ASCENDING')

        if cpu:
            sorted_tensor = sorted_tensor.numpy()
            indices = indices.numpy()

        return {
            "sorted_values": sorted_tensor,
            "indices": indices
        }

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "dim": -1,
        "descending": False,
        "stable": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    np.testing.assert_almost_equal(torch_result["sorted_values"], tf_result["sorted_values"], decimal=6)
    np.testing.assert_almost_equal(torch_result["indices"], tf_result["indices"], decimal=6)
    
    if np.allclose(torch_result["sorted_values"], tf_result["sorted_values"]) and np.allclose(torch_result["indices"], tf_result["indices"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()