import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    downscale_factor = input_dict.get("downscale_factor", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()

    pixel_unshuffle = torch.nn.PixelUnshuffle(downscale_factor=downscale_factor)
    result = pixel_unshuffle(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        downscale_factor = input_dict.get("downscale_factor", 1)

        input_shape = tf.shape(input_tensor)
        rank = len(input_tensor.shape)

        if rank == 3:
            input_tensor = tf.expand_dims(input_tensor, axis=0)

        batch_size = tf.shape(input_tensor)[0]
        height = tf.shape(input_tensor)[1]
        width = tf.shape(input_tensor)[2]
        channels = tf.shape(input_tensor)[3]

        new_height = height * downscale_factor
        new_width = width * downscale_factor
        new_channels = channels // (downscale_factor * downscale_factor)

        x = tf.reshape(input_tensor, [batch_size, height // downscale_factor, width // downscale_factor, downscale_factor, downscale_factor, channels])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        result = tf.reshape(x, [batch_size, new_height, new_width, new_channels])

        if rank == 3:
            result = tf.squeeze(result, axis=0)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 4, 4, 32).astype(np.float32),
        "downscale_factor": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()