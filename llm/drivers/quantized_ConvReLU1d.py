import numpy as np
import torch

def torch_version(input_dict, cpu=True):
    from torch.nn.intrinsic.quantized import ConvReLU1d

    input_tensor = torch.tensor(input_dict['input'])
    weight = torch.tensor(input_dict['weight'])
    bias = torch.tensor(input_dict['bias']) if 'bias' in input_dict else None
    stride = input_dict.get('stride', 1)
    padding = input_dict.get('padding', 0)
    dilation = input_dict.get('dilation', 1)
    groups = input_dict.get('groups', 1)
    padding_mode = input_dict.get('padding_mode', 'zeros')

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    model = ConvReLU1d(
        in_channels=input_dict['in_channels'],
        out_channels=input_dict['out_channels'],
        kernel_size=input_dict['kernel_size'],
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        padding_mode=padding_mode,
    )
    with torch.no_grad():
        model.weight.data.copy_(weight.data)
        if bias is not None:
            model.bias.data.copy_(bias.data)

    model.eval()
    with torch.no_grad():
        result = model(input_tensor)

    if not cpu:
        result = result.cpu()

    return {'result': result.numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict['input'])
        weight = tf.constant(input_dict['weight'])
        bias = tf.constant(input_dict['bias']) if 'bias' in input_dict else None
        stride = input_dict.get('stride', 1)
        padding = input_dict.get('padding', 0)
        dilation = input_dict.get('dilation', 1)
        groups = input_dict.get('groups', 1)
        padding_mode = input_dict.get('padding_mode', 'zeros')

        input_tensor = tf.expand_dims(input_tensor, axis=0)

        if len(weight.shape) == 2:
             weight = tf.expand_dims(tf.transpose(weight, perm=[1, 0]), axis=0)
        elif len(weight.shape) == 3:
            weight = tf.transpose(weight, perm=[0, 2, 1])

        if padding_mode != 'zeros':
            raise ValueError(f"Padding mode {padding_mode} not supported in TensorFlow.")

        result = tf.nn.conv1d(
            input=input_tensor,
            filters=weight,
            stride=stride,
            padding='VALID',
            dilation_rate=dilation
        )

        if bias is not None:
            result = tf.nn.bias_add(result, tf.constant(bias))

        result = tf.nn.relu(result)
        result = tf.squeeze(result, axis=0)
        result = result.numpy()

    return {'result': result}


def main():
    A_TOL = 0.01

    input_data = {
        'input': np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float32),
        'weight': np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]], dtype=np.float32),
        'bias': np.array([0.1, 0.2], dtype=np.float32),
        'in_channels': 1,
        'out_channels': 2,
        'kernel_size': 3,
        'stride': 1,
        'padding': 0,
        'dilation': 1,
        'groups': 1,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()