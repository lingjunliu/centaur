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

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        upscale_factor = input_dict["upscale_factor"]

        input_shape = input_tensor.shape
        rank = len(input_shape)

        if rank == 4:
          batch_size, height, width, channels = input_shape
          channels = channels // (upscale_factor * upscale_factor)
          x = tf.reshape(input_tensor, [batch_size, height, width, upscale_factor, upscale_factor, channels])
          x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
          x = tf.reshape(x, [batch_size, height * upscale_factor, width * upscale_factor, channels])
          result = x
        elif rank == 3:
          height, width, channels = input_shape
          channels = channels // (upscale_factor * upscale_factor)
          x = tf.reshape(input_tensor, [height, width, upscale_factor, upscale_factor, channels])
          x = tf.transpose(x, [0, 2, 1, 3, 4])
          x = tf.reshape(x, [height * upscale_factor, width * upscale_factor, channels])
          result = x

        elif rank == 5:
            batch_size, depth, height, width, channels = input_shape
            channels = channels // (upscale_factor * upscale_factor)
            x = tf.reshape(input_tensor, [batch_size, depth, height, width, upscale_factor, upscale_factor, channels])
            x = tf.transpose(x, [0, 1, 2, 4, 3, 5, 6])
            x = tf.reshape(x, [batch_size, depth, height * upscale_factor, width * upscale_factor, channels])
            result = x
        elif rank == 6:
            batch_size, time, depth, height, width, channels = input_shape
            channels = channels // (upscale_factor * upscale_factor)
            x = tf.reshape(input_tensor, [batch_size, time, depth, height, width, upscale_factor, upscale_factor, channels])
            x = tf.transpose(x, [0, 1, 2, 3, 5, 4, 6, 7])
            x = tf.reshape(x, [batch_size, time, depth, height * upscale_factor, width * upscale_factor, channels])
            result = x
        else:
            raise ValueError(f"Input tensor must have rank 3, 4, 5, or 6, but has rank {rank}")

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[[1, 2, 3, 4]]]], dtype=np.float32),
        "upscale_factor": 2
    }
    input_data["input"] = np.repeat(input_data["input"], 4, axis=-1)

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()