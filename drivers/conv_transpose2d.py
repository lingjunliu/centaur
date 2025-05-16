
def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    
    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    weight_tensor = torch.tensor(input["weight"])
    # bias_tensor = torch.tensor(input.get("bias", None)) if input.get("bias", None) is not None else None
    stride = input.get("stride", 1)
    padding = input.get("padding", 0)
    # output_padding = input.get("output_padding", 0)
    # groups = input.get("groups", 1)
    # dilation = input.get("dilation", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()
        if weight_tensor is not None:
            weight_tensor = weight_tensor.cuda()
        # if bias_tensor is not None:
        #     bias_tensor = bias_tensor.cuda()

    # Apply to torch.nn.functional.conv_transpose2d
    # output = torch.nn.functional.conv_transpose2d(
    #     input_tensor, weight_tensor, bias=bias_tensor, stride=stride,
    #     padding=padding, output_padding=output_padding, groups=groups, dilation=dilation
    # )
    output = torch.nn.functional.conv_transpose2d(
        input_tensor, weight_tensor, stride=stride, padding=padding
    )

    if not cpu:
        output = output.cpu()

    return {"conv_transpose2d_output": output.detach().numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        weight_tensor = tf.constant(input["weight"])
        bias_tensor = tf.constant(input.get("bias", None)) if input.get("bias", None) is not None else None
        stride = input.get("stride", 1)
        padding = input.get("padding", 0)
        output_padding = input.get("output_padding", 0)
        groups = input.get("groups", 1)
        dilation = input.get("dilation", 1)
        
        input_tensor = tf.transpose(input_tensor, [0, 2, 3, 1])
        weight_tensor = tf.transpose(weight_tensor, [2, 3, 1, 0])  # Adjust weight shape for TensorFlow format

        batch_size = tf.shape(input_tensor)[0]
        input_height = input_tensor.shape[1]
        input_width = input_tensor.shape[2]
        filter_height = weight_tensor.shape[0]
        filter_width = weight_tensor.shape[1]
        
        out_channels = weight_tensor.shape[3]  # TensorFlow expects the number of filters in the weight tensor to match output channels
        
        h_out = (input_height - 1) * stride - 2 * padding + dilation * (filter_height - 1) + 1 + output_padding
        w_out = (input_width - 1) * stride - 2 * padding + dilation * (filter_width - 1) + 1 + output_padding
        output_shape = [batch_size, h_out, w_out, out_channels]

        # Apply to TensorFlow equivalent
        output = tf.nn.conv2d_transpose(
            input_tensor, weight_tensor, output_shape=output_shape, strides=[1, stride, stride, 1],
            padding='VALID' if padding == 0 else 'SAME', dilations=[1, dilation, dilation, 1]
        )

        # Add bias if available
        if bias_tensor is not None:
            output = tf.nn.bias_add(output, bias_tensor)

        output = tf.transpose(output, [0, 3, 1, 2])
        return {"conv_transpose2d_output": output.numpy()}

def main():
    import numpy as np
    # Example input
    input_data = {
        "input": np.random.randn(1, 3, 32, 32).astype(np.float32),  # Batch size 1, 3 channels, 32x32 image
        "weight": np.random.randn(3, 3, 3, 3).astype(np.float32),  # 3x3 filter size, 3 input channels, 3 output channels
        "bias": np.random.randn(3).astype(np.float32),  # Bias for each output channel
        "stride": 1,
        "padding": 0,
        "output_padding": 0,
        "groups": 1,
        "dilation": 1,
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Use numpy for comparison to ensure exact precision matching
    if np.allclose(torch_result["conv_transpose2d_output"], tf_result["conv_transpose2d_output"], atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()