import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    k = input.get("k", 1)
    dims = input.get("dims", [0, 1])

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply to torch.rot90
    rotated = torch.rot90(input_tensor, k=k, dims=dims)

    if not cpu:
        rotated = rotated.cpu()

    return {"rot90_result": rotated.numpy()}

def tensorflow_rotate_90(input_tensor, k, dims):
    import tensorflow as tf
    # Correct the rotation logic to ensure it matches the tensor's rank
    k = k % 4  # Only 4 unique rotations (0°, 90°, 180°, 270°)

    for _ in range(k):
        # Rotate 90 degrees by reversing the axis and then transposing
        input_tensor = tf.transpose(tf.reverse(input_tensor, axis=[dims[1]]), perm=[*range(dims[0]), dims[1], dims[0], *range(dims[1]+1, len(input_tensor.shape))])
        
    return input_tensor

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        k = input.get("k", 1)
        dims = input.get("dims", [0, 1])

        # Apply to TensorFlow equivalent
        rotated = tensorflow_rotate_90(input_tensor, k, dims)

        return {"rot90_result": rotated.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.float32),  # 3D tensor example
        "k": 1,  # Number of 90-degree rotations
        "dims": [1, 2]  # Rotation dimensions
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare the results
    if np.array_equal(torch_result["rot90_result"], tf_result["rot90_result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()