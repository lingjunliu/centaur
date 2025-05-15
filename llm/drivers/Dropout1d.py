import numpy as np
import tensorflow as tf
import torch

def torch_version(input_dict, cpu=True):
    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict.get("p", 0.5)
    inplace = input_dict.get("inplace", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    m = torch.nn.Dropout1d(p=p, inplace=inplace)
    output = m(input_tensor)

    if not cpu:
        output = output.cpu()

    return {"result": output.numpy()}

def tensorflow_version(input_dict, cpu=True):
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.convert_to_tensor(input_dict["input"], dtype=tf.float32)
        p = input_dict.get("p", 0.5)

        shape = tf.shape(input_tensor)
        if len(shape) == 3:
            N, C, L = shape[0], shape[1], shape[2]
            random_tensor = tf.random.uniform(shape=[N, C, 1], minval=0.0, maxval=1.0, dtype=tf.float32)
            binary_mask = tf.cast(random_tensor >= p, dtype=tf.float32)
            output_tensor = tf.multiply(input_tensor, binary_mask) / (1 - p)
        elif len(shape) == 2:
            C, L = shape[0], shape[1]
            random_tensor = tf.random.uniform(shape=[1, C, 1], minval=0.0, maxval=1.0, dtype=tf.float32)
            binary_mask = tf.cast(random_tensor >= p, dtype=tf.float32)
            output_tensor = tf.multiply(input_tensor[None, ...], binary_mask) / (1 - p)
            output_tensor = output_tensor[0, ...]
        else:
            raise ValueError("Input tensor must have shape (N, C, L) or (C, L)")

        output = output_tensor.numpy()
    return {"result": output}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.randn(20, 16, 32).astype(np.float32),
        "p": 0.2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()