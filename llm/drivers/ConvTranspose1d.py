import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    input_tensor = torch.tensor(input_dict["input"], dtype=torch.float32)
    weight = torch.tensor(input_dict["weight"], dtype=torch.float32)
    bias = torch.tensor(input_dict.get("bias", np.zeros(input_dict["weight"].shape[0])), dtype=torch.float32) if "bias" in input_dict else None
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    output_padding = input_dict.get("output_padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    input_tensor = input_tensor.unsqueeze(0).unsqueeze(1)

    conv_transpose1d = nn.ConvTranspose1d(
        in_channels=input_dict["weight"].shape[1],
        out_channels=input_dict["weight"].shape[0],
        kernel_size=input_dict["weight"].shape[2],
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        bias=(bias is not None),
    )

    conv_transpose1d.weight.data = weight
    if bias is not None:
        conv_transpose1d.bias.data = bias

    result = conv_transpose1d(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.squeeze(0).squeeze(0).numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias = tf.constant(input_dict.get("bias", np.zeros(input_dict["weight"].shape[0])), dtype=tf.float32) if "bias" in input_dict else None
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        output_padding = input_dict.get("output_padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)

        input_shape = input_tensor.shape
        in_channels = weight.shape[1]
        out_channels = weight.shape[0]
        kernel_size = weight.shape[2]

        if groups != 1:
            raise NotImplementedError("Groups > 1 not implemented for ConvTranspose1D in TensorFlow.")

        input_tensor = tf.expand_dims(input_tensor, axis=0)
        input_tensor = tf.expand_dims(input_tensor, axis=-1)

        output_shape = (
            input_shape[0],
            (input_shape[0] - 1) * stride
            - 2 * padding
            + dilation * (kernel_size - 1)
            + output_padding
            + 1,
            out_channels,
        )

        output_shape_tensor = tf.constant([1, output_shape, out_channels])

        weight_reshaped = tf.transpose(weight, perm=[2, 1, 0])
        weight_reshaped = tf.expand_dims(weight_reshaped, axis=0)

        result = tf.nn.conv1d_transpose(
            input=input_tensor,
            filters=weight_reshaped,
            output_shape=output_shape_tensor,
            strides=stride,
            padding="VALID",
            data_format="NWC",
            dilation_rate=dilation,
        )

        if bias is not None:
            result = tf.nn.bias_add(result, bias, data_format="NWC")

        result = tf.squeeze(result, axis=0)
        result = result.numpy()
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "weight": np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32).reshape(2,1,2),
        "bias": np.array([0.1, 0.2], dtype=np.float32),
        "stride": 2,
        "padding": 1,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()