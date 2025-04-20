import numpy as np

def torch_pow_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    exponent = input["exponent"]
    if isinstance(exponent, list):
        exponent_tensor = torch.tensor(exponent)
    else:
        exponent_tensor = exponent

    # Apply torch.pow
    result = torch.pow(input_tensor, exponent_tensor)

    if not cpu:
        result = result.cpu()

    return {"pow_result": result.numpy()}

def tensorflow_pow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        exponent = input["exponent"]
        if isinstance(exponent, list):
            exponent_tensor = tf.constant(exponent)
        else:
            exponent_tensor = tf.constant(exponent)

        # Apply tf pow function
        result = tf.math.pow(input_tensor, exponent_tensor)

        return {"pow_result": result.numpy()}

def main():
    # Example input 1 (scalar exponent)
    input_data_1 = {
        "input": np.array([0.5, 2.0, 3.0, 4.0], dtype=np.float32),
        "exponent": 2
    }

    # Torch example
    torch_result_1 = torch_pow_version(input_data_1)
    print("Torch result 1:", torch_result_1)

    # TensorFlow example
    tf_result_1 = tensorflow_pow_version(input_data_1)
    print("TensorFlow result 1:", tf_result_1)
    
    assert np.allclose(torch_result_1["pow_result"], tf_result_1["pow_result"]), "Results do not match"
    print("equal")

    # Example input 2 (tensor exponent)
    input_data_2 = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "exponent": [1, 2, 3, 4]
    }

    # Torch example
    torch_result_2 = torch_pow_version(input_data_2)
    print("Torch result 2:", torch_result_2)

    # TensorFlow example
    tf_result_2 = tensorflow_pow_version(input_data_2)
    print("TensorFlow result 2:", tf_result_2)

    assert np.allclose(torch_result_2["pow_result"], tf_result_2["pow_result"]), "Results do not match"
    print("equal")

if __name__ == "__main__":
    main()