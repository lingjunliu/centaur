import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["other"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        other_tensor = other_tensor.cuda()

    # Apply torch.copysign
    result = torch.copysign(input_tensor, other_tensor)

    if not cpu:
        result = result.cpu()

    return {"copysign_result": result.numpy()}

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

        # Apply TensorFlow equivalent
        sign_tensor = tf.math.sign(other_tensor)
        abs_input = tf.math.abs(input_tensor)
        result = abs_input * sign_tensor

        return {"copysign_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[-1.2, 3.5, -2.5], [1.3, -4.7, 3.1]], dtype=np.float32),
        "other": np.array([[0.8, -0.3, 0.5], [-0.2, 0.6, -0.7]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_np_result = torch_result["copysign_result"]
    tf_np_result = tf_result["copysign_result"]

    if np.allclose(torch_np_result, tf_np_result, atol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()