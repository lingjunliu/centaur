import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input_dict["input"])

    # Move tensor to GPU if required
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply torch.trunc
    result_tensor = torch.trunc(input_tensor)

    # Move the result to CPU if it was on GPU
    if not cpu:
        result_tensor = result_tensor.cpu()

    return {"trunc_result": result_tensor.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input_dict["input"])

        # Apply TensorFlow equivalent of trunc (tf.floor is used)
        result_tensor = tf.sign(input_tensor) * tf.floor(tf.abs(input_tensor))

        return {"trunc_result": result_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([3.4742, 0.5466, -0.8008, -0.9079], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert equality by converting results to numpy arrays
    if np.allclose(torch_result["trunc_result"], tf_result["trunc_result"], atol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()