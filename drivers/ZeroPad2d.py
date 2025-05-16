import numpy as np

def torch_version_zero_pad_2d(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    padding = input["padding"]

    # Apply ZeroPad2d
    pad = torch.nn.ZeroPad2d(padding)
    padded_tensor = pad(input_tensor)

    if not cpu:
        padded_tensor = padded_tensor.cpu()

    return padded_tensor.numpy()

def tensorflow_version_zero_pad_2d(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        padding = input["padding"]
        
        # Convert padding to the form needed by tf.pad
        if isinstance(padding, int):
            tf_padding = [[0, 0], [0, 0], [padding, padding], [padding, padding]]
        elif len(padding) == 4:
            tf_padding = [[0, 0], [0, 0], [padding[2], padding[3]], [padding[0], padding[1]]]
        else:
            raise ValueError("Padding must be an int or a 4-tuple")

        # Apply padding
        padded_tensor = tf.pad(input_tensor, tf_padding, mode='CONSTANT', constant_values=0.0)

        return padded_tensor.numpy()

def main():
    # Example input
    input_data = {
        "input": np.random.randn(1, 1, 3, 3).astype(np.float32),
        "padding": (2, 2, 2, 2),  # Padding values
    }

    # Torch example
    torch_result = torch_version_zero_pad_2d(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version_zero_pad_2d(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    if np.array_equal(torch_result, tf_result):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()