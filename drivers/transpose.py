import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    dim0 = input["dim0"]
    dim1 = input["dim1"]

    # Apply torch.transpose
    result = torch.transpose(input_tensor, dim0, dim1)

    if not cpu:
        result = result.cpu()

    return {"transpose_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        dim0 = input["dim0"]
        dim1 = input["dim1"]

        # Apply tf.transpose
        perm = list(range(len(input_tensor.shape)))
        perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        result = tf.transpose(input_tensor, perm=perm)

        return {"transpose_result": result.numpy()}

def main():
    # Example input data
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "dim0": 0,
        "dim1": 1
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result["transpose_result"])

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result["transpose_result"])

    # Compare results
    if np.array_equal(torch_result["transpose_result"], tf_result["transpose_result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()