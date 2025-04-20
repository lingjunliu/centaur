import numpy as np

def torch_polygamma_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    n_val = input["n"]
    input_tensor = torch.tensor(input["input"])

    # Apply torch.polygamma
    result_tensor = torch.polygamma(n_val, input_tensor)

    if not cpu:
        result_tensor = result_tensor.cpu()

    return {"polygamma_result": result_tensor.numpy()}

def tensorflow_polygamma_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        n_val = input["n"]
        input_tensor = tf.constant(input["input"])

        # Apply TensorFlow equivalent using tf.math.polygamma
        result_tensor = tf.math.polygamma(n_val, input_tensor)

        return {"polygamma_result": result_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "n": 2,
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
    }

    # Torch example
    torch_result = torch_polygamma_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_polygamma_version(input_data)
    print("TensorFlow result:", tf_result)

    # Comparing the results
    assert np.allclose(torch_result["polygamma_result"], tf_result["polygamma_result"]), "Results are not equal"
    print("equal")

if __name__ == "__main__":
    main()