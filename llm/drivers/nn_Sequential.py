import numpy as np
from collections import OrderedDict

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn

    args = input_dict.get("args", [])
    if isinstance(args, list):
        modules = []
        for i, module_params in enumerate(args):
            module_type = module_params["type"]
            if module_type == "Conv2d":
                modules.append(nn.Conv2d(
                    in_channels=module_params["in_channels"],
                    out_channels=module_params["out_channels"],
                    kernel_size=module_params["kernel_size"],
                    stride=module_params.get("stride", 1),
                    padding=module_params.get("padding", 0),
                    dilation=module_params.get("dilation", 1),
                    groups=module_params.get("groups", 1),
                    bias=module_params.get("bias", True),
                    padding_mode=module_params.get("padding_mode", 'zeros')
                ))
            elif module_type == "ReLU":
                modules.append(nn.ReLU())
    elif isinstance(args, OrderedDict):
        modules = []
        for name, module_params in args.items():
            module_type = module_params["type"]
            if module_type == "Conv2d":
                modules.append(nn.Conv2d(
                    in_channels=module_params["in_channels"],
                    out_channels=module_params["out_channels"],
                    kernel_size=module_params["kernel_size"],
                    stride=module_params.get("stride", 1),
                    padding=module_params.get("padding", 0),
                    dilation=module_params.get("dilation", 1),
                    groups=module_params.get("groups", 1),
                    bias=module_params.get("bias", True),
                    padding_mode=module_params.get("padding_mode", 'zeros')
                ))
            elif module_type == "ReLU":
                modules.append(nn.ReLU())
    else:
        modules = []

    model = nn.Sequential(*modules)

    input_tensor = torch.tensor(input_dict["input"].reshape(1, 1, 28, 28).float())

    if not cpu:
        model = model.cuda()
        input_tensor = input_tensor.cuda()

    result = model(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy().flatten()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        args = input_dict.get("args", [])
        layers = []
        first_conv = True
        for module_params in args:
            module_type = module_params["type"]
            if module_type == "Conv2d":
                input_shape = (28, 28, 1) if first_conv else None
                first_conv = False
                conv_layer = tf.keras.layers.Conv2D(
                    filters=module_params["out_channels"],
                    kernel_size=module_params["kernel_size"],
                    strides=module_params.get("stride", 1),
                    padding="same" if module_params.get("padding", 0) != 0 else "valid",
                    dilation_rate=module_params.get("dilation", 1),
                    groups=module_params.get("groups", 1),
                    use_bias=module_params.get("bias", True),
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    input_shape=input_shape
                )
                layers.append(conv_layer)
            elif module_type == "ReLU":
                layers.append(tf.keras.layers.ReLU())

        model = tf.keras.Sequential(layers)

        input_tensor = tf.constant(input_dict["input"].reshape(1, 28, 28, 1), dtype=tf.float32)

        result = model(input_tensor)

        result = result.numpy()

    return {"result": result.flatten()}


def main():
    A_TOL = 1e-2
    input_data = {
        "input": np.random.rand(28 * 28).astype(np.float32),
        "args": [
            {
                "type": "Conv2d",
                "in_channels": 1,
                "out_channels": 20,
                "kernel_size": 5
            },
            {
                "type": "ReLU"
            },
            {
                "type": "Conv2d",
                "in_channels": 20,
                "out_channels": 64,
                "kernel_size": 5
            },
            {
                "type": "ReLU"
            }
        ]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()