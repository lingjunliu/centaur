import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    dim = input.get("dim", 0)

    # Apply torch.unbind
    result = torch.unbind(input_tensor, dim=dim)

    if not cpu:
        result = [tensor.cpu() for tensor in result]

    return {"unbind_result": [tensor.numpy() for tensor in result]}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        dim = input.get("dim", 0)

        # Apply TensorFlow equivalent
        result = tf.unstack(input_tensor, axis=dim)

        return {"unbind_result": [tensor.numpy() for tensor in result]}

def main():
    # Example input
    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "dim": 0,
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    torch_np_result = np.array(torch_result["unbind_result"])
    tf_np_result = np.array(tf_result["unbind_result"])

    if np.array_equal(torch_np_result, tf_np_result):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()