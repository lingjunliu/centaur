import numpy as np

def calculate_output_shape(input_shape, weight_shape, stride, padding, output_padding, dilation):
    N, C_in, D_in, H_in, W_in = input_shape
    C_out, _, kernel_d, kernel_h, kernel_w = weight_shape

    # Calculate the dimensions of the output using the formula
    D_out = (D_in - 1) * stride - 2 * padding + dilation * (kernel_d - 1) + output_padding + 1
    H_out = (H_in - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1
    W_out = (W_in - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1

    return (N, C_out, D_out, H_out, W_out)

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    weight_tensor = torch.tensor(input["weight"])
    if input.get("bias", None) is not None:
        bias_tensor = torch.tensor(input.get("bias", None))
    else:
        bias_tensor = None
    stride = input.get("stride", 1)
    padding = input.get("padding", 0)
    output_padding = input.get("output_padding", 0)
    groups = input.get("groups", 1)
    dilation = input.get("dilation", 1)

    # Apply to torch.nn.functional.conv_transpose3d
    output = torch.nn.functional.conv_transpose3d(
        input=input_tensor, weight=weight_tensor, bias=bias_tensor,
        stride=stride, padding=padding, output_padding=output_padding,
        groups=groups, dilation=dilation
    )

    if not cpu:
        output = output.cpu()

    return {"conv_transpose3d_output": output.detach().numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(np.transpose(input["input"], (0, 2, 3, 4, 1)))
        weight_tensor = tf.constant(np.transpose(input["weight"], (2, 3, 4, 1, 0)))
        if input.get("bias", None) is not None:
            bias_tensor = tf.constant(input.get("bias", None))
        else:
            bias_tensor = None
        stride = input.get("stride", 1)
        padding = 'SAME' if input.get("padding", 0) > 0 else 'VALID'
        output_padding = input.get("output_padding", 0)
        groups = input.get("groups", 1)
        dilation = input.get("dilation", 1)

        input_shape = input["input"].shape
        weight_shape = input["weight"].shape

        # Infer the explicit output shape
        output_shape = calculate_output_shape(input_shape, weight_shape, stride, input.get("padding", 0), output_padding, dilation)
        output_shape = (output_shape[0], output_shape[2], output_shape[3], output_shape[4], output_shape[1])

        # TensorFlow conv3d_transpose
        output = tf.nn.conv3d_transpose(
            input=input_tensor, filters=weight_tensor,
            output_shape=output_shape, strides=[1, stride, stride, stride, 1], padding=padding
        )

        if bias_tensor is not None:
            output = tf.nn.bias_add(output, bias_tensor)

        # Transpose back to 'NCDHW'
        output = tf.transpose(output, (0, 4, 1, 2, 3))

        return {"conv_transpose3d_output": output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.rand(2, 16, 8, 8, 8).astype(np.float32),  # Batch, Channels, Depth, Height, Width
        "weight": np.random.rand(16, 16, 3, 3, 3).astype(np.float32),  # Out Channels, In Channels, Depth, Height, Width
        "bias": np.random.rand(16).astype(np.float32),  # Change to None if no bias is needed
        "stride": 2,
        "padding": 1,  # Becomes 'SAME' in TensorFlow if > 0
        "output_padding": 0,
        "groups": 1,
        "dilation": 1
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    if np.allclose(torch_result["conv_transpose3d_output"], tf_result["conv_transpose3d_output"], atol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()