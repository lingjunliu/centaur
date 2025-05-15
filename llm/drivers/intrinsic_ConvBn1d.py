import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"])
    running_mean = torch.tensor(input_dict["running_mean"])
    running_var = torch.tensor(input_dict["running_var"])

    eps = input_dict.get("eps", 1e-05)
    momentum = input_dict.get("momentum", 0.1)
    affine = input_dict.get("affine", True)
    track_running_stats = input_dict.get("track_running_stats", True)
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    padding_mode = input_dict.get("padding_mode", 'zeros')
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        bias = bias.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()

    conv1d = nn.Conv1d(
        input_dict["in_channels"],
        input_dict["out_channels"],
        input_dict["kernel_size"],
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=affine,
        padding_mode=padding_mode
    )
    bn1d = nn.BatchNorm1d(input_dict["out_channels"], eps, momentum, affine, track_running_stats)
    conv1d.weight.data = weight.reshape(input_dict["out_channels"], input_dict["in_channels"] // groups, input_dict["kernel_size"])
    conv1d.bias.data = bias
    bn1d.running_mean = running_mean
    bn1d.running_var = running_var
    bn1d.weight.data = torch.ones(input_dict["out_channels"]) if affine else None
    bn1d.bias.data = torch.zeros(input_dict["out_channels"]) if affine else None
    conv1d.eval()
    bn1d.eval()

    with torch.no_grad():
        result = bn1d(conv1d(input_tensor.unsqueeze(0).unsqueeze(0)).squeeze(0))

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias = tf.constant(input_dict["bias"], dtype=tf.float32)
        running_mean = tf.constant(input_dict["running_mean"], dtype=tf.float32)
        running_var = tf.constant(input_dict["running_var"], dtype=tf.float32)

        eps = input_dict.get("eps", 1e-05)
        momentum = input_dict.get("momentum", 0.1)
        affine = input_dict.get("affine", True)
        track_running_stats = input_dict.get("track_running_stats", True)
        stride=input_dict.get("stride", 1)
        padding=input_dict.get("padding", 0)
        dilation=input_dict.get("dilation", 1)
        groups=input_dict.get("groups", 1)
        
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        input_tensor = tf.expand_dims(input_tensor, axis=2)

        weight = tf.reshape(weight, (input_dict["kernel_size"], 1, input_dict["in_channels"], input_dict["out_channels"]))
        weight = tf.transpose(weight, perm=[3, 2, 0, 1])
        
        conv = tf.nn.conv2d(input_tensor, weight, strides=[1, stride, 1, 1], padding='VALID', dilations=[1, dilation, 1, 1])

        if affine:
          scale = tf.nn.batch_normalization(
              conv,
              mean=running_mean,
              variance=running_var,
              offset=bias,
              scale=tf.ones(input_dict["out_channels"], dtype=tf.float32),
              variance_epsilon=eps
          )
        else:
          scale = tf.nn.batch_normalization(
              conv,
              mean=running_mean,
              variance=running_var,
              offset=tf.zeros(input_dict["out_channels"], dtype=tf.float32),
              scale=tf.ones(input_dict["out_channels"], dtype=tf.float32),
              variance_epsilon=eps
          )

        result = tf.squeeze(scale, axis=[0,2]).numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        "weight": np.array([0.5, 0.6, 0.7], dtype=np.float32),
        "bias": np.array([0.1], dtype=np.float32),
        "running_mean": np.array([0.2], dtype=np.float32),
        "running_var": np.array([0.3], dtype=np.float32),
        "in_channels": 1,
        "out_channels": 1,
        "kernel_size": 3,
        "stride": 1,
        "padding": 0
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()