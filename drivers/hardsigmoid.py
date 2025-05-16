import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    # Apply the Hardsigmoid function
    hard_sigmoid = torch.nn.Hardsigmoid()
    output_tensor = hard_sigmoid(input_tensor)

    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"hardsigmoid_output": output_tensor.numpy()}

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
        
        # Define the custom Hardsigmoid function
        def hardsigmoid(x):
            return tf.where(x <= -3, 0.0, tf.where(x >= 3, 1.0, x / 6 + 1 / 2))

        # Apply the function
        output_tensor = hardsigmoid(input_tensor)

        return {"hardsigmoid_output": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[-3, -2, -1, 0, 1, 2, 3, 4]], dtype=np.float32),
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    assert np.allclose(torch_result["hardsigmoid_output"], tf_result["hardsigmoid_output"]), "Results are not equal"
    print("equal")

if __name__ == "__main__":
    main()