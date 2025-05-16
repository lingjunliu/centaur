import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

def torch_version(input_dict, cpu=True):
    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict.get("p", 0.5)
    inplace = input_dict.get("inplace", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    m = torch.nn.Dropout3d(p=p, inplace=inplace)
    output = m(input_tensor)

    if not cpu:
        output = output.cpu()

    return {"result": output.numpy()}

def tensorflow_version(input_dict, cpu=True):
    input_tensor = input_dict["input"]
    p = input_dict.get("p", 0.5)

    input_shape = input_tensor.shape

    if len(input_shape) == 5:
        N, C, D, H, W = input_shape
        input_tensor_tf = tf.convert_to_tensor(input_tensor, dtype=tf.float32)

        random_tensor = tf.random.uniform(shape=[N, C], minval=0, maxval=1, dtype=tf.float32)
        binary_mask = tf.cast(random_tensor > p, dtype=tf.float32)
        binary_mask = tf.reshape(binary_mask, [N, C, 1, 1, 1])
        output_tensor = input_tensor_tf * binary_mask / (1 - p)

    else:
        C, D, H, W = input_shape
        input_tensor_tf = tf.convert_to_tensor(input_tensor, dtype=tf.float32)

        random_tensor = tf.random.uniform(shape=[C], minval=0, maxval=1, dtype=tf.float32)
        binary_mask = tf.cast(random_tensor > p, dtype=tf.float32)
        binary_mask = tf.reshape(binary_mask, [C, 1, 1, 1])
        output_tensor = input_tensor_tf * binary_mask / (1 - p)

    return {"result": output_tensor.numpy()}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 4, 5, 6).astype(np.float32),
        "p": 0.3
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data2 = {
        "input": np.random.rand(3, 4, 5, 6).astype(np.float32),
        "p": 0.3
    }

    torch_result2 = torch_version(input_data2)
    tf_result2 = tensorflow_version(input_data2)

    assert np.allclose(torch_result2["result"], tf_result2["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()