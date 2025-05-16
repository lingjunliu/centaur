import numpy as np

# Assuming the set_seed function is defined in src.setseed

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    
    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["other"])

    # Compute element-wise maximum
    result_tensor = torch.maximum(input_tensor, other_tensor)

    if not cpu:
        result_tensor = result_tensor.cpu()

    return {"maximum_tensor": result_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        other_tensor = tf.constant(input["other"])

        # Compute element-wise maximum
        result_tensor = tf.math.maximum(input_tensor, other_tensor)

        return {"maximum_tensor": result_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[1, 2, -1], [3, 0, 4]], dtype=np.float32),
        "other": np.array([[3, 0, 4], [1, 5, -3]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data, cpu=True)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data, cpu=True)
    print("TensorFlow result:", tf_result)

    # Compare results
    np.testing.assert_array_equal(torch_result["maximum_tensor"], tf_result["maximum_tensor"])
    print("equal")

if __name__ == "__main__":
    main()