import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = input_dict.get("bias", None)
    if bias is not None:
      bias = torch.tensor(bias)
    stride = input_dict.get("stride", (1,1))
    padding = input_dict.get("padding", (2,2))
    dilation = input_dict.get("dilation", (1,1))
    groups = input_dict.get("groups", 1)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()
    
    if len(padding) == 2:
        result = torch.mkldnn_convolution(input_tensor, weight, bias, stride, (padding[0], padding[1]), dilation, groups)
    else:
        result = torch.mkldnn_convolution(input_tensor, weight, bias, stride, padding, dilation, groups)
    
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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        bias = input_dict.get("bias", None)
        if bias is not None:
            bias = tf.constant(bias, dtype=tf.float32)
        stride = input_dict.get("stride", (1,1))
        padding = input_dict.get("padding", (2,2))
        dilation = input_dict.get("dilation", (1,1))
        groups = input_dict.get("groups", 1)

        input_tensor = tf.reshape(input_tensor, [1, input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[0]])
        weight = tf.reshape(weight, [weight.shape[2], weight.shape[3], weight.shape[1], weight.shape[0]])
        
        padding_tf = 'VALID'
        if padding != (0,0):
          padding_tf = 'SAME'
        
        strides_tf = [1, stride[0], stride[1], 1]
        dilations_tf = [1, dilation[0], dilation[1], 1]

        result = tf.nn.conv2d(input_tensor, weight, strides=strides_tf, padding=padding_tf, dilations=dilations_tf)
        if bias is not None:
            result = tf.nn.bias_add(result, bias)
        
        result = result.numpy()
        result = np.transpose(result[0], (2, 0, 1))
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(3, 32, 32).astype(np.float32),
        "weight": np.random.rand(16, 3, 5, 5).astype(np.float32),
        "bias": np.random.rand(16).astype(np.float32),
        "stride": (1, 1),
        "padding": (2, 2),
        "dilation": (1, 1),
        "groups": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()