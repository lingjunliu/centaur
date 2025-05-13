import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from torch.nn.utils import fuse_conv_bn_eval

    conv = input_dict["conv"]
    bn = input_dict["bn"]

    conv.eval()
    bn.eval()

    if not cpu:
        conv = conv.cuda()
        bn = bn.cuda()

    fused_conv = fuse_conv_bn_eval(conv, bn)

    if not cpu:
        fused_conv = fused_conv.cpu()

    if hasattr(fused_conv, 'weight'):
        return {"fused_conv": fused_conv.weight.detach().numpy()}
    else:
        return {"fused_conv": fused_conv.detach().numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    import numpy as np

    conv = input_dict["conv"]
    bn = input_dict["bn"]

    conv_weights_np = conv.weight.detach().numpy()
    bn_running_mean_np = bn.running_mean.detach().numpy()
    bn_running_var_np = bn.running_var.detach().numpy()
    bn_weight_np = bn.weight.detach().numpy()
    bn_bias_np = bn.bias.detach().numpy()
    bn_eps = bn.eps

    if len(conv_weights_np.shape) == 4:
        w_shape = conv_weights_np.shape
        w = conv_weights_np.transpose((2, 3, 1, 0))

        gamma = bn_weight_np
        beta = bn_bias_np
        mean = bn_running_mean_np
        var = bn_running_var_np
        epsilon = bn_eps

        std = np.sqrt(var + epsilon)
        new_w = gamma / std * w
        new_b = beta - mean * gamma / std

        new_w = new_w.transpose((3, 2, 0, 1))
        fused_conv_tf = new_w
    else:
        w = conv_weights_np

        gamma = bn_weight_np
        beta = bn_bias_np
        mean = bn_running_mean_np
        var = bn_running_var_np
        epsilon = bn_eps

        std = np.sqrt(var + epsilon)
        new_w = gamma / std * w
        new_b = beta - mean * gamma / std

        fused_conv_tf = new_w

    return {"fused_conv": fused_conv_tf}


def main():
    A_TOL = 0.01
    import torch
    import torch.nn as nn

    # Example input
    conv = nn.Conv2d(3, 16, kernel_size=3)
    bn = nn.BatchNorm2d(16)
    input_data = {
        "conv": conv,
        "bn": bn
    }

    # Torch example
    torch_result = torch_version(input_data)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)

    # Assert to see if they are equal
    assert np.allclose(torch_result["fused_conv"], tf_result["fused_conv"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()