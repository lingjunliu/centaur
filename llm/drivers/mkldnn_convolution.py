import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn.functional as F

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None
    stride = input_dict.get("stride", (1, 1))
    padding = input_dict.get("padding", (0, 0))
    dilation = input_dict.get("dilation", (1, 1))
    groups = input_dict.get("groups", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    result = F.conv2d(
        input_tensor,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
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
        weight = tf.constant(input_dict["weight"])
        bias = tf.constant(input_dict["bias"]) if "bias" in input_dict else None
        stride = input_dict.get("stride", (1, 1))
        padding = input_dict.get("padding", (0, 0))
        dilation = input_dict.get("dilation", (1, 1))
        groups = input_dict.get("groups", 1)

        stride_list = [1, stride[0], stride[1], 1]
        dilation_list = [1, dilation[0], dilation[1], 1]

        if isinstance(padding, int):
            padding_type = "VALID"
        else:
            padding_type = "VALID"
            
        input_shape = input_tensor.shape
        weight_shape = weight.shape

        if len(input_shape) == 4 and len(weight_shape) == 4:
            
            in_channels = int(input_shape[-1])
            out_channels = int(weight_shape[0])
            kernel_height = int(weight_shape[1])
            kernel_width = int(weight_shape[2])
            
            if groups == 1:
                result = tf.nn.conv2d(
                    input_tensor,
                    weight,
                    strides=stride_list,
                    padding=padding_type,
                    dilations=dilation_list,
                )
            else:
                group_size = in_channels // groups
                kernel_size = weight_shape[2]

                input_split = tf.split(input_tensor, num_or_size_splits=groups, axis=3)
                
                # Corrected Weight split
                weight_reshaped = tf.reshape(weight, [groups, out_channels // groups, kernel_height, kernel_width, in_channels // groups])
                weight_split = tf.unstack(weight_reshaped, axis=0)
                
                convolved_groups = []
                for i in range(groups):
                    convolved_group = tf.nn.conv2d(
                        input_split[i],
                        weight_split[i],
                        strides=stride_list,
                        padding=padding_type,
                        dilations=dilation_list,
                    )
                    convolved_groups.append(convolved_group)

                result = tf.concat(convolved_groups, axis=3)

            if bias is not None:
                result = tf.nn.bias_add(result, bias)

        else:
            raise ValueError("Only 2D convolution is supported")

        result = result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "weight": np.random.rand(16, 3, 5, 5).astype(np.float32),
        "bias": np.random.rand(16).astype(np.float32),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "groups": 1,
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()