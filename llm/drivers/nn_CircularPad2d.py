import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict["padding"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    m = nn.CircularPad2d(padding)
    result = m(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    padding = input_dict["padding"]

    def circular_pad_2d(tensor, padding):
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        
        padding_left, padding_right, padding_top, padding_bottom = padding

        shape = tf.shape(tensor)
        height = shape[-2]
        width = shape[-1]

        top_pad = tf.concat([tensor[..., height-padding_top:, :], tensor[..., :height, :]], axis=-2)
        bottom_pad = tf.concat([tensor[..., :padding_bottom, :], tensor[..., :height, :]], axis=-2)

        left_pad = tf.concat([tensor[..., :, width-padding_left:], tensor[..., :, :width]], axis=-1)
        right_pad = tf.concat([tensor[..., :, :padding_right], tensor[..., :, :width]], axis=-1)

        padded = tensor

        if padding_top > 0:
            top_pad = top_pad[..., :padding_top, :]
            padded = tf.concat([top_pad, padded], axis=-2)

        if padding_bottom > 0:
            bottom_pad = bottom_pad[..., :padding_bottom, :]
            padded = tf.concat([padded, bottom_pad], axis=-2)
        
        if padding_left > 0:
            left_pad = left_pad[..., :, :padding_left]
            padded = tf.concat([left_pad, padded], axis=-1)

        if padding_right > 0:
            right_pad = right_pad[..., :, :padding_right]
            padded = tf.concat([padded, right_pad], axis=-1)
            
        return padded

    result = circular_pad_2d(input_tensor, padding)
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