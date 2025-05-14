import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply torch.t
    result = torch.t(input_tensor)

    return {"t_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        if tf.rank(input_tensor) == 2:
            result = tf.transpose(input_tensor)
        else:
            result = input_tensor

        return {"t_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert the outputs are equal
    np.testing.assert_allclose(torch_result["t_result"], tf_result["t_result"], rtol=1e-5)
    print("equal")

if __name__ == "__main__":
    main()