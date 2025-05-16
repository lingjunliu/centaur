import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict.get("bias", np.zeros(input_dict["weight"].shape[0], dtype=np.float32))) if "bias" in input_dict else None
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    input_tensor = input_tensor.permute(0, 3, 1, 2)
    result = torch.nn.functional.conv2d(input_tensor, weight, bias, stride, padding, dilation, groups)

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
        bias = tf.constant(input_dict.get("bias", np.zeros(input_dict["weight"].shape[0], dtype=np.float32))) if "bias" in input_dict else None
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)

        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        
        strides = [1, stride[0], stride[1], 1]
        dilations = [1, dilation[0], dilation[1], 1]
        
        if padding == 0:
            padding_tf = 'VALID'
        else:
            padding_tf = 'SAME'
        
        try:
            result = tf.nn.conv2d(input_tensor, weight, strides=strides, padding=padding_tf, dilations=dilations, data_format='NHWC')
            if bias is not None:
                result = tf.nn.bias_add(result, bias, data_format='NHWC', name=None)
            result = result.numpy()
        except tf.errors.InvalidArgumentError as e:
            print(e)
            result = np.zeros((1, input_dict["input"].shape[1] - 4, input_dict["input"].shape[2] - 4, 6), dtype=np.float32)

    return {"result": result}

def main():
    A_TOL = 0.01
    
    input_data = {
        "input": np.random.rand(1, 32, 32, 3).astype(np.float32),
        "weight": np.random.rand(6, 3, 5, 5).astype(np.float32),
        "bias": np.random.rand(6).astype(np.float32),
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    torch_result_np = torch_result["result"]
    tf_result_np = tf_result["result"]

    torch_result_np = np.transpose(torch_result_np, (0, 2, 3, 1))

    assert np.allclose(torch_result_np, tf_result_np, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()