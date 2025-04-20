import numpy as np


def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["other"])
    alpha = input.get("alpha", 1.0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        other_tensor = other_tensor.cuda()

    # Apply torch.sub
    result = torch.sub(input_tensor, other_tensor, alpha=alpha)

    if not cpu:
        result = result.cpu()

    return {"sub_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        other_tensor = tf.constant(input["other"])
        alpha = input.get("alpha", 1.0)

        # Apply TensorFlow equivalent
        other_scaled = other_tensor * alpha
        result = input_tensor - other_scaled

        return {"sub_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "other": np.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]], dtype=np.float32),
        "alpha": 1.0
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert and print if they are equal or not
    assert np.allclose(torch_result["sub_result"], tf_result["sub_result"]), "Results are not equal"
    print("equal")

if __name__ == "__main__":
    main()