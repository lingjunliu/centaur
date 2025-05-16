import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    
    # Unpack input dictionary
    input_tensor = torch.tensor(input["tensor"])
    dim = input["dim"]

    # Apply torch.unsqueeze
    result_tensor = torch.unsqueeze(input_tensor, dim)

    if not cpu:
        result_tensor = result_tensor.cpu()

    return {"unsqueeze_result": result_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["tensor"])
        dim = input["dim"]

        # Apply tf.expand_dims
        result_tensor = tf.expand_dims(input_tensor, axis=dim)

        return {"unsqueeze_result": result_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "tensor": np.array([1, 2, 3, 4], dtype=np.float32),
        "dim": 0
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_output = torch_result["unsqueeze_result"]
    tf_output = tf_result["unsqueeze_result"]

    if np.array_equal(torch_output, tf_output):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()