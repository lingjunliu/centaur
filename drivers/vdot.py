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

    # Apply to torch.vdot
    result = torch.vdot(input_tensor, other_tensor)

    if not cpu:
        result = result.cpu()

    return {"vdot_result": result.numpy()}

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

        # Apply to TensorFlow equivalent
        result = tf.tensordot(input_tensor, other_tensor, axes=1)

        return {"vdot_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([2.0, 3.0], dtype=np.float32),
        "other": np.array([2.0, 1.0], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare the results
    torch_val = torch_result["vdot_result"]
    tf_val = tf_result["vdot_result"]
    if np.isclose(torch_val, tf_val):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()