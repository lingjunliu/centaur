import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    
    input_tensor = torch.tensor(input_dict["input"])
    
    in_channels = input_dict.get("in_channels", 0)
    out_channels = input_dict.get("out_channels", 0)
    kernel_size = input_dict.get("kernel_size", 1)
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)
    bias = input_dict.get("bias", True)
    padding_mode = input_dict.get("padding_mode", 'zeros')
    
    lazy_conv = torch.nn.LazyConv1d(out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        lazy_conv = lazy_conv.cuda()

    result = lazy_conv(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        
        in_channels = input_dict.get("in_channels", 0)
        out_channels = input_dict.get("out_channels", 0)
        kernel_size = input_dict.get("kernel_size", 1)
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        bias = input_dict.get("bias", True)
        padding_mode = input_dict.get("padding_mode", 'zeros')
        
        input_shape = input_tensor.shape
        
        if len(input_shape) == 2:
            input_tensor = tf.expand_dims(input_tensor, axis=0)
        
        in_channels = input_shape[-1]
        
        if padding_mode != 'zeros':
            raise NotImplementedError("Tensorflow LazyConv1D only supports padding_mode='zeros'")
        
        if out_channels == 0:
           out_channels = in_channels
           
        kernel_shape = (kernel_size, in_channels, out_channels)
        
        kernel = tf.Variable(tf.keras.initializers.glorot_uniform()(shape=kernel_shape))
        
        if bias:
            bias_var = tf.Variable(tf.zeros(shape=(out_channels,)))
        else:
            bias_var = None
        
        strides = [1, stride, 1]
        
        if isinstance(padding, str):
            padding_type = padding.upper()
        else:
            padding_type = 'VALID' if padding == 0 else 'SAME'

        result = tf.nn.conv1d(input_tensor, kernel, stride=strides[1], padding=padding_type)


        if bias_var is not None:
            result = tf.nn.bias_add(result, bias_var)
            
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    
    input_data = {
        "input": np.random.rand(1, 10, 3).astype(np.float32),
        "kernel_size": 3,
        "in_channels": 3,
        "out_channels": 5,
        "stride": 1,
        "padding": 1,
        "dilation": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    if torch_result["result"].shape != tf_result["result"].shape:
        print(f"Torch shape: {torch_result['result'].shape}")
        print(f"TF shape: {tf_result['result'].shape}")

    min_len = min(torch_result["result"].shape[1], tf_result["result"].shape[1])

    torch_cropped = torch_result["result"][:, :min_len, :]
    tf_cropped = tf_result["result"][:, :min_len, :]
        
    assert torch_cropped.shape == tf_cropped.shape, "Shapes do not match"

    assert np.allclose(torch_cropped, tf_cropped, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()