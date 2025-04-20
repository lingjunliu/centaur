import numpy as np

def torch_version_elu(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"]).clone()  # Using clone to mimic in-place operation
    alpha = input.get("alpha", 1.0)

    # Apply ELU activation function
    result = torch.nn.functional.elu_(input_tensor, alpha=alpha)

    if not cpu:
        result = result.cpu()

    return {"elu_result": result.numpy()}

def tensorflow_version_elu(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        alpha = input.get("alpha", 1.0)

        # Apply ELU activation function
        result = tf.nn.elu(input_tensor)

        if alpha != 1.0:
            result = tf.where(result >= 0, result, alpha * (tf.exp(result) - 1))

        return {"elu_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, -0.3, 0.8], [-0.2, 0.6, -0.9]], dtype=np.float32),
        "alpha": 1.0
    }

    # Torch example
    torch_result = torch_version_elu(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version_elu(input_data)
    print("TensorFlow result:", tf_result)

    # Use numpy to compare the results
    if np.allclose(torch_result["elu_result"], tf_result["elu_result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()