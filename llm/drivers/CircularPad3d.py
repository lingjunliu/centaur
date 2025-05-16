import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict.get("padding", (1, 1, 1))

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.functional.pad(input_tensor, (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0]), mode='circular')

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    padding = input_dict.get("padding", (1, 1, 1))

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        pad_depth, pad_height, pad_width = padding

        input_shape = tf.shape(input_tensor)
        depth = input_shape[2]
        height = input_shape[3]
        width = input_shape[4]

        depth_pad = tf.concat([input_tensor[:, :, -pad_depth:, :, :], input_tensor, input_tensor[:, :, :pad_depth, :, :]], axis=2)
        height_pad = tf.concat([depth_pad[:, :, :, -pad_height:, :], depth_pad, depth_pad[:, :, :, :pad_height, :]], axis=3)
        width_pad = tf.concat([height_pad[:, :, :, :, -pad_width:], height_pad, height_pad[:, :, :, :, :pad_width]], axis=4)

        result = width_pad.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 1, 3, 3, 3).astype(np.float32),
        "padding": (1, 1, 1)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()