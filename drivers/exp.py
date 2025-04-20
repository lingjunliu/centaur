import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    # Apply torch.exp
    result = torch.exp(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"exp_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # Apply tf.exp
        result = tf.exp(input_tensor)

        return {"exp_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.0, np.log(2.0)], [1.0, np.log(3.0)]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    assert np.allclose(torch_result["exp_result"], tf_result["exp_result"]), "Results differ!"
    print("equal" if np.allclose(torch_result["exp_result"], tf_result["exp_result"]) else "not equal")

if __name__ == "__main__":
    main()