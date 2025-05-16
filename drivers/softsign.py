import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply torch.nn.functional.softsign
    output = torch.nn.functional.softsign(input_tensor)

    if not cpu:
        output = output.cpu()
    
    return {"softsign_output": output.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string="/cpu:0"
    else:
        device_string="/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # Apply TensorFlow equivalent
        output = tf.nn.softsign(input_tensor)

        return {"softsign_output": output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[1.0, -2.0, 0.0], [3.0, 0.5, -0.5]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    np.testing.assert_almost_equal(
        torch_result["softsign_output"], tf_result["softsign_output"], decimal=5
    )
    print("Results are equal")

if __name__ == "__main__":
    main()