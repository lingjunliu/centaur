import numpy as np
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

def torch_version(input_dict, cpu=True):
    input_tensor = torch.tensor(input_dict["input"], dtype=torch.float32)
    padding = input_dict["padding"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    circular_pad = torch.nn.CircularPad1d(padding)
    result = circular_pad(input_tensor)

    if not cpu:
        result = result.cpu()

    result = result.squeeze()

    return {"result": result.numpy()}


def tensorflow_version(input_dict, cpu=True):
    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    padding = input_dict["padding"]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        input_tensor = tf.expand_dims(input_tensor, axis=0)

        padding_left = padding if isinstance(padding, int) else padding[0]
        padding_right = padding if isinstance(padding, int) else padding[1] if len(padding) > 1 else padding[0]
        
        input_shape = tf.shape(input_tensor)[-1]

        padded_left = tf.gather(input_tensor, tf.range(input_shape - padding_left, input_shape), axis=2)
        padded_right = tf.gather(input_tensor, tf.range(padding_right), axis=2)

        padded_tensor = tf.concat([padded_left, input_tensor, padded_right], axis=2)

        result = tf.squeeze(padded_tensor).numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "padding": (2, 3)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "padding": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()