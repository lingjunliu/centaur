import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    inplace = input.get("inplace", False)
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply torch.nn.functional.relu
    result = torch.nn.functional.relu(input_tensor, inplace=inplace)

    # Move result back to CPU if necessary
    if not cpu:
        result = result.cpu()

    return {"relu_result": result.numpy()}

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

        # Apply tf.nn.relu
        result = tf.nn.relu(input_tensor)

        return {"relu_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[-1.0, 0.0, 1.0], [0.5, -0.5, 2.0]], dtype=np.float32),
        "inplace": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    np.testing.assert_allclose(torch_result["relu_result"], tf_result["relu_result"], rtol=1e-5, atol=1e-8)
    print("equal")

if __name__ == "__main__":
    main()