import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict.get("padding")

    if not cpu:
        input_tensor = input_tensor.cuda()

    m = nn.CircularPad3d(padding)
    output = m(input_tensor)

    if not cpu:
        output = output.cpu()

    return {"result": output.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    padding = input_dict.get("padding")

    if isinstance(padding, int):
        padding_left = padding_right = padding_top = padding_bottom = padding_front = padding_back = padding
    else:
        padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back = padding

    input_tensor_shape = tf.shape(input_tensor)
    rank = len(input_tensor.shape)
    
    if rank == 5:
        n, c, din, hin, win = input_tensor_shape[0], input_tensor_shape[1], input_tensor_shape[2], input_tensor_shape[3], input_tensor_shape[4]
    elif rank == 4:
        c, din, hin, win = input_tensor_shape[0], input_tensor_shape[1], input_tensor_shape[2], input_tensor_shape[3]
    else:
        raise ValueError("Input tensor must be 4D or 5D")

    def circular_pad_depth(tensor, padding_front, padding_back):
        if padding_front > 0:
            pad_front = tf.concat([tensor[..., -padding_front:, :, :], tensor[..., :padding_front, :, :]], axis=-3)
            tensor = tf.concat([pad_front, tensor], axis=-3)
        if padding_back > 0:
            pad_back = tf.concat([tensor[..., :padding_back, :, :], tensor[..., -padding_back:, :, :]], axis=-3)
            tensor = tf.concat([tensor, pad_back], axis=-3)
        return tensor

    def circular_pad_height(tensor, padding_top, padding_bottom):
        if padding_top > 0:
            pad_top = tf.concat([tensor[..., -padding_top:, :], tensor[..., :padding_top, :]], axis=-2)
            tensor = tf.concat([pad_top, tensor], axis=-2)
        if padding_bottom > 0:
            pad_bottom = tf.concat([tensor[..., :padding_bottom, :], tensor[..., -padding_bottom:, :]], axis=-2)
            tensor = tf.concat([tensor, pad_bottom], axis=-2)
        return tensor

    def circular_pad_width(tensor, padding_left, padding_right):
        if padding_left > 0:
            pad_left = tf.concat([tensor[..., -padding_left:], tensor[..., :padding_left]], axis=-1)
            tensor = tf.concat([pad_left, tensor], axis=-1)
        if padding_right > 0:
            pad_right = tf.concat([tensor[..., :padding_right], tensor[..., -padding_right:]], axis=-1)
            tensor = tf.concat([tensor, pad_right], axis=-1)
        return tensor
    
    if rank == 5:
        pad_tensor = tf.pad(input_tensor, [[0,0],[0,0],[padding_front, padding_back],[padding_top, padding_bottom],[padding_left, padding_right]], mode="REFLECT")
        padded_tensor = circular_pad_depth(input_tensor, padding_front, padding_back)
        padded_tensor = circular_pad_height(padded_tensor, padding_top, padding_bottom)
        padded_tensor = circular_pad_width(padded_tensor, padding_left, padding_right)

    elif rank == 4:
        padded_tensor = circular_pad_depth(input_tensor, padding_front, padding_back)
        padded_tensor = circular_pad_height(padded_tensor, padding_top, padding_bottom)
        padded_tensor = circular_pad_width(padded_tensor, padding_left, padding_right)
    else:
        raise ValueError("Input tensor must be 4D or 5D")

    return {"result": padded_tensor.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 4, 5, 6).astype(np.float32),
        "padding": (1, 1, 2, 2, 3, 3)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(3, 4, 5, 6).astype(np.float32),
        "padding": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    

    print("Success")

if __name__ == "__main__":
    main()