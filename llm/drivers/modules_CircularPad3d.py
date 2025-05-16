import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict["padding"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    pad3d = torch.nn.CircularPad3d(padding)
    result = pad3d(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    padding = input_dict["padding"]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        def circular_pad_3d(tensor, padding):
            if isinstance(padding, int):
                padding = [padding] * 3
            elif len(padding) == 1:
                padding = padding * 3
            elif len(padding) != 3:
                raise ValueError("Padding must be an int or a tuple/list of length 1 or 3")

            input_shape = tf.shape(tensor)
            z_pad, y_pad, x_pad = padding

            def get_circular_indices(pad_size, input_size):
                indices = tf.range(-pad_size, input_size + pad_size)
                indices = tf.math.floormod(indices, input_size)
                return tf.cast(indices, tf.int32)

            z_indices = get_circular_indices(z_pad, input_shape[1])
            y_indices = get_circular_indices(y_pad, input_shape[2])
            x_indices = get_circular_indices(x_pad, input_shape[3])

            padded_tensor = tf.gather(tensor, tf.reshape(z_indices, [-1, 1, 1, 1, 1]), axis=1)
            padded_tensor = tf.gather(padded_tensor, tf.reshape(y_indices, [1, -1, 1, 1, 1]), axis=2)
            padded_tensor = tf.gather(padded_tensor, tf.reshape(x_indices, [1, 1, -1, 1, 1]), axis=3)

            return padded_tensor

        result = circular_pad_3d(input_tensor, padding)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 5, 5, 5, 1).astype(np.float32),
        "padding": (2, 1, 3)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(1, 3, 4, 5, 1).astype(np.float32),
        "padding": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()