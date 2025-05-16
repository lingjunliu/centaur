import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    values_tensor = torch.tensor(input["values"])

    # Apply to torch.heaviside
    result = torch.heaviside(input_tensor, values_tensor)

    if not cpu:
        result = result.cpu()

    return {"heaviside_result": result.numpy()}

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
        values_tensor = tf.constant(input["values"])

        # Create the Heaviside step function logic in TensorFlow
        result = tf.where(input_tensor < 0, tf.zeros_like(input_tensor), 
                          tf.where(input_tensor > 0, tf.ones_like(input_tensor), values_tensor))

        return {"heaviside_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([-1.5, 0, 2.0], dtype=np.float32),
        "values": np.array([0.5, -1.0, 3.5], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare the results
    torch_result_array = np.array(torch_result["heaviside_result"])
    tf_result_array = np.array(tf_result["heaviside_result"])

    if np.allclose(torch_result_array, tf_result_array):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()