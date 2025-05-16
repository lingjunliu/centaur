import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    upscale_factor = input_dict["upscale_factor"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    pixel_shuffle = torch.nn.PixelShuffle(upscale_factor)

    if not cpu:
        pixel_shuffle = pixel_shuffle.cuda()

    result = pixel_shuffle(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    upscale_factor = input_dict["upscale_factor"]

    input_shape = input_tensor.shape
    if len(input_shape) == 4:
        b, h, w, c = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    elif len(input_shape) == 3:
        h, w, c = input_shape[0], input_shape[1], input_shape[2]
        b = 1
        input_tensor = tf.expand_dims(input_tensor, axis=0)
    elif len(input_shape) == 2:
        h, w = input_shape[0], input_shape[1]
        c = 1
        b = 1
        input_tensor = tf.reshape(input_tensor, (b, h, w, c))
    else:
        raise ValueError("Input tensor must have 2, 3, or 4 dimensions.")

    new_h = h * upscale_factor
    new_w = w * upscale_factor
    r = upscale_factor

    output_tensor = tf.reshape(input_tensor, [b, h, w, r, r, c // (r * r)])
    output_tensor = tf.transpose(output_tensor, [0, 1, 3, 2, 4, 5])
    output_tensor = tf.reshape(output_tensor, [b, new_h, new_w, c // (r * r)])

    if len(input_shape) == 3:
        output_tensor = tf.squeeze(output_tensor, axis=0)
    elif len(input_shape) == 2:
        output_tensor = tf.squeeze(output_tensor, axis=0)
        output_tensor = tf.squeeze(output_tensor, axis=-1)

    return {"result": output_tensor.numpy()}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[[[[1, 2, 3, 4, 5, 6, 7, 8]]], [[[9, 10, 11, 12, 13, 14, 15, 16]]]]], dtype=np.float32),
        "upscale_factor": 2
    }

    input_data["input"] = np.array([[[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]]], dtype=np.float32)
    input_data["input"] = input_data["input"].reshape((1,1,1,16))

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()