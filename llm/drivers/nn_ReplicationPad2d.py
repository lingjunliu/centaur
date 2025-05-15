import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict.get("padding", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    m = nn.ReplicationPad2d(padding)
    result = m(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    padding = input_dict.get("padding", 0)

    if isinstance(padding, int):
        padding_left = padding_right = padding_top = padding_bottom = padding
    else:
        padding_left, padding_right, padding_top, padding_bottom = padding

    input_shape = input_tensor.shape
    rank = len(input_shape)

    if rank == 3:
        input_tensor = tf.expand_dims(input_tensor, axis=0)
    elif rank == 2:
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        input_tensor = tf.expand_dims(input_tensor, axis=0)
    elif rank == 1:
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        input_tensor = tf.expand_dims(input_tensor, axis=0)

    mode = 'REFLECT'
    if mode == 'REFLECT':
        mode = 'SYMMETRIC'

    if rank == 4:
        paddings = [[0, 0], [0, 0], [padding_top, padding_bottom], [padding_left, padding_right]]
    elif rank == 3:
        paddings = [[0, 0], [padding_top, padding_bottom], [padding_left, padding_right]]
    elif rank == 2:
        paddings = [[0, 0], [0, 0], [padding_top, padding_bottom], [padding_left, padding_right]]
    elif rank == 1:
        paddings = [[0, 0], [0, 0], [0, 0], [padding_top, padding_bottom], [padding_left, padding_right]]
    else:
        paddings = [[padding_top, padding_bottom], [padding_left, padding_right]]

    result = tf.pad(input_tensor, paddings, mode='CONSTANT', constant_values=input_tensor[0, ... , 0].numpy().item())

    if rank == 3:
        result = tf.squeeze(result, axis=0)
    elif rank == 2:
        result = tf.squeeze(result, axis=0)
        result = tf.squeeze(result, axis=0)
    elif rank == 1:
        result = tf.squeeze(result, axis=0)
        result = tf.squeeze(result, axis=0)
        result = tf.squeeze(result, axis=0)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3),
        "padding": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3),
        "padding": (1, 1, 2, 0)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()