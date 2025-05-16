import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    obj = input["obj"]

    # Apply torch.is_tensor
    is_tensor = torch.is_tensor(obj)

    return {"is_tensor": is_tensor}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        obj = input["obj"]

        # TensorFlow equivalent: Check if obj is a tensor
        is_tensor = tf.is_tensor(obj)

        return {"is_tensor": is_tensor}

def main():
    # Example input
    input_data = {
        "obj": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32)  # Ensure input is float type
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert and compare the results
    torch_result_tensor = torch.tensor(torch_result["is_tensor"])
    tf_result_tensor = tf.convert_to_tensor(tf_result["is_tensor"])

    if torch_result_tensor.numpy() == tf_result_tensor.numpy():
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()