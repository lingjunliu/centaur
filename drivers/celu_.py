import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    alpha = input.get("alpha", 1.0)
    inplace = input.get("inplace", False)

    # Apply torch.nn.functional.celu
    output = torch.nn.functional.celu(input_tensor, alpha=alpha, inplace=inplace)

    if not cpu:
        output = output.cpu()

    return {"celu_output": output.numpy()}

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
        alpha = input.get("alpha", 1.0)

        # Implement the CELU function
        positive_part = tf.maximum(0.0, input_tensor)
        negative_part = tf.minimum(0.0, alpha * (tf.exp(input_tensor / alpha) - 1))
        output = positive_part + negative_part

        return {"celu_output": output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, -0.3, 0.8], [-0.2, 0.6, -0.9]], dtype=np.float32),
        "alpha": 1.0,
        "inplace": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_output = torch_result["celu_output"]
    tf_output = tf_result["celu_output"]

    if np.allclose(torch_output, tf_output, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()