import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    padding = input["padding"]

    # Move tensor to device if not CPU
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply torch.nn.ReplicationPad1d
    m = torch.nn.ReplicationPad1d(padding)
    output = m(input_tensor)

    if not cpu:
        output = output.cpu()

    return { 'output': output.numpy() } # Convert output to numpy array for comparison

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        padding = input["padding"]

        # Create a custom 1D replication pad function in TensorFlow
        if isinstance(padding, int):
            padding_left = padding_right = padding
        elif isinstance(padding, tuple) and len(padding) == 2:
            padding_left, padding_right = padding
        else:
            raise ValueError("Padding must be an int or a tuple of 2 integers")

        # Perform the replication padding manually
        left_pad = tf.repeat(input_tensor[:, :, :1], padding_left, axis=2)  # Replicate the left boundary
        right_pad = tf.repeat(input_tensor[:, :, -1:], padding_right, axis=2)  # Replicate the right boundary

        output = tf.concat([left_pad, input_tensor, right_pad], axis=2)

        return { 'output': output.numpy() }  # Convert output to numpy array for comparison

def main():
    # Example input
    input_data = {
        "input": np.arange(8, dtype=np.float32).reshape(1, 2, 4),  # Example 1D tensor input
        "padding": (3, 1)  # Example padding
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results using numpy array comparison
    if np.array_equal(torch_result['output'], tf_result['output']):
        print("equal")
    else:
        print("not equal")


if __name__ == "__main__":
    main()