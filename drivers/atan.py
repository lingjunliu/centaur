import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    # Apply torch.atan
    output_tensor = torch.atan(input_tensor)
    
    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"atan_output": output_tensor.numpy()}

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

        # Apply TensorFlow equivalent (tf.math.atan)
        output_tensor = tf.math.atan(input_tensor)

        return {"atan_output": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([0.2341, 0.2539, -0.6256, -0.6448], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert equality considering numerical precision
    assert np.allclose(torch_result["atan_output"], tf_result["atan_output"], atol=1e-7), "not equal"
    print("equal")

if __name__ == "__main__":
    main()