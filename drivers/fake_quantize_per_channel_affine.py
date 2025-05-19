import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    scale = torch.tensor(input_dict["scale"])
    zero_point = torch.tensor(input_dict["zero_point"]).int()
    quant_min = input_dict.get("quant_min", 0)
    quant_max = input_dict.get("quant_max", 255)
    axis = input_dict.get("axis", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        scale = scale.cuda()
        zero_point = zero_point.cuda()

    result = torch.fake_quantize_per_channel_affine(
        input_tensor, scale, zero_point, axis=axis, quant_min=quant_min, quant_max=quant_max
    )

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
        scale = tf.constant(input_dict["scale"])
        zero_point = tf.constant(input_dict["zero_point"])
        quant_min = input_dict.get("quant_min", 0)
        quant_max = input_dict.get("quant_max", 255)
        axis = input_dict.get("axis", 0)
        
        input_shape = tf.shape(input_tensor)
        num_channels = input_shape[axis]

        scale = tf.reshape(scale, [num_channels if i == axis else 1 for i in range(len(input_shape))])
        zero_point = tf.reshape(zero_point, [num_channels if i == axis else 1 for i in range(len(input_shape))])

        scale = tf.cast(scale, input_tensor.dtype)
        zero_point = tf.cast(zero_point, input_tensor.dtype)

        quantized = tf.clip_by_value(
            tf.round(input_tensor / scale + zero_point), quant_min, quant_max
        )
        dequantized = (quantized - zero_point) * scale

        result = dequantized.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array(
            [
                [[-0.7383, -1.4055, -0.1998], [-0.6346, 0.9323, 0.3695]],
                [[1.6277, 0.4571, 0.3145], [1.2481, 0.1812, 0.6342]],
            ],
            dtype=np.float32,
        ),
        "scale": np.array([0.0564, 0.0388, 0.0450], dtype=np.float32),
        "zero_point": np.array([21, 19, 22], dtype=np.int32),
        "quant_min": 0,
        "quant_max": 255,
        "axis": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(
        torch_result["result"], tf_result["result"], atol=A_TOL
    ), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()