import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    
    input_tensor = torch.tensor(input_dict["input"])
    conv = torch.nn.Conv1d(input_dict["in_channels"], input_dict["out_channels"], input_dict["kernel_size"], stride=input_dict.get("stride", 1), padding=input_dict.get("padding", 0), dilation=input_dict.get("dilation", 1), groups=input_dict.get("groups", 1), bias=input_dict.get("bias", True), padding_mode=input_dict.get("padding_mode", 'zeros'))
    bn = torch.nn.BatchNorm1d(input_dict["out_channels"])
    relu = torch.nn.ReLU()

    conv.weight = torch.nn.Parameter(torch.tensor(input_dict["conv_weight"]))
    if input_dict.get("bias", True):
        conv.bias = torch.nn.Parameter(torch.tensor(input_dict["conv_bias"]))
    bn.weight = torch.nn.Parameter(torch.tensor(input_dict["bn_weight"]))
    bn.bias = torch.nn.Parameter(torch.tensor(input_dict["bn_bias"]))
    bn.running_mean = torch.tensor(np.random.rand(input_dict["out_channels"]).astype(np.float32))
    bn.running_var = torch.tensor(np.random.rand(input_dict["out_channels"]).astype(np.float32))
    bn.eps = input_dict["bn_eps"]
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        conv = conv.cuda()
        bn = bn.cuda()
        relu = relu.cuda()
        conv.weight = torch.nn.Parameter(conv.weight.cuda())
        if input_dict.get("bias", True):
            conv.bias = torch.nn.Parameter(conv.bias.cuda())
        bn.weight = torch.nn.Parameter(bn.weight.cuda())
        bn.bias = torch.nn.Parameter(bn.bias.cuda())
        bn.running_mean = bn.running_mean.cuda()
        bn.running_var = bn.running_var.cuda()
        
    model = torch.nn.intrinsic.ConvBnReLU1d(conv, bn, relu)

    input_tensor = input_tensor.unsqueeze(0)
    result = model(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.squeeze(0).numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        conv_weight = tf.constant(input_dict["conv_weight"], dtype=tf.float32)
        if input_dict.get("bias", True):
            conv_bias = tf.constant(input_dict["conv_bias"], dtype=tf.float32)
        bn_weight = tf.constant(input_dict["bn_weight"], dtype=tf.float32)
        bn_bias = tf.constant(input_dict["bn_bias"], dtype=tf.float32)
        bn_running_mean = tf.constant(input_dict["bn_running_mean"], dtype=tf.float32)
        bn_running_var = tf.constant(input_dict["bn_running_var"], dtype=tf.float32)
        bn_eps = input_dict["bn_eps"]
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        
        x = tf.reshape(input_tensor, [1, input_tensor.shape[0], 1])

        padding_total = (input_dict["kernel_size"] - 1) * input_dict.get("dilation", 1)
        padding_before = padding_total // 2
        padding_after = padding_total - padding_before
        x_padded = tf.pad(x, [[0, 0], [padding_before, padding_after], [0, 0]], "CONSTANT")
        
        conv_weight = tf.transpose(conv_weight, perm=[2, 0, 1])
        
        conv = tf.nn.conv1d(x_padded, conv_weight, stride=stride, padding="VALID", dilation_rate=dilation)
        
        if input_dict.get("bias", True):
          conv = tf.nn.bias_add(conv, conv_bias)

        gamma = tf.reshape(bn_weight, [1, -1, 1])
        beta = tf.reshape(bn_bias, [1, -1, 1])
        mean = tf.reshape(bn_running_mean, [1, -1, 1])
        variance = tf.reshape(bn_running_var, [1, -1, 1])

        bn = tf.nn.batch_normalization(conv, mean, variance, beta, gamma, bn_eps)
        relu = tf.nn.relu(bn)
        result = relu

        result = tf.reshape(result, [input_dict["out_channels"], -1])
        result = tf.transpose(result)
        
        result = result.numpy()
        
    return {"result": result}

def main():
    A_TOL = 0.01
    input_len = 5
    out_channels = 2
    kernel_size = 3
    input_data = {
        "input": np.random.rand(input_len).astype(np.float32),
        "in_channels": 1,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "bias": True,
        "padding_mode": 'zeros',
        "conv_weight": np.random.rand(out_channels, 1, kernel_size).astype(np.float32),
        "conv_bias": np.random.rand(out_channels).astype(np.float32),
        "bn_weight": np.random.rand(out_channels).astype(np.float32),
        "bn_bias": np.random.rand(out_channels).astype(np.float32),
        "bn_running_mean": np.random.rand(out_channels).astype(np.float32),
        "bn_running_var": np.random.rand(out_channels).astype(np.float32),
        "bn_eps": 1e-05
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()