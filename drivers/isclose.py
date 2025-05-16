import numpy as np

def torch_isclose(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["other"])
    rtol = input.get("rtol", 1e-05)
    atol = input.get("atol", 1e-08)
    equal_nan = input.get("equal_nan", False)

    # Apply torch.isclose
    result = torch.isclose(input_tensor, other_tensor, rtol=rtol, atol=atol, equal_nan=equal_nan)

    if not cpu:
        result = result.cpu()

    return {"isclose_result": result.numpy()}

def tensorflow_isclose(input, cpu=True):
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
        rtol = input.get("rtol", 1e-05)
        atol = input.get("atol", 1e-08)
        equal_nan = input.get("equal_nan", False)

        # Compute the TensorFlow isclose equivalent
        abs_diff = tf.abs(input_tensor - other_tensor)
        acceptable_error = atol + rtol * tf.abs(other_tensor)
        is_close = tf.less_equal(abs_diff, acceptable_error)

        if equal_nan:
            both_nan_condition = tf.logical_and(tf.is_nan(input_tensor), tf.is_nan(other_tensor))
            is_close = tf.logical_or(is_close, both_nan_condition)

        return {"isclose_result": is_close.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "other": np.array([1.0 + 1e-10, 2.0, 4.0], dtype=np.float32),  # Ensure other is float type
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }

    # Torch example
    torch_result = torch_isclose(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_isclose(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_isclose_result = torch_result["isclose_result"]
    tf_isclose_result = tf_result["isclose_result"]

    if np.array_equal(torch_isclose_result, tf_isclose_result):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()