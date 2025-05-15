import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from torch.nn.intrinsic import ConvReLU2d
    import torch.nn as nn

    module_type = input_dict['module_type']
    
    if module_type == 'ConvReLU2d':
        in_channels = input_dict["in_channels"]
        out_channels = input_dict["out_channels"]
        kernel_size = input_dict["kernel_size"]
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        padding_mode = input_dict.get("padding_mode", 'zeros')
        bias = input_dict.get("bias", True)

        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, bias=bias)
        mod = ConvReLU2d(conv)
        
        input_tensor = torch.tensor(input_dict["input"])

        if not cpu:
            mod = mod.cuda()
            input_tensor = input_tensor.cuda()

        result = mod(input_tensor)

        if not cpu:
            result = result.cpu()
        
        return {"result": result.numpy()}

    elif module_type == 'Conv2d':
        in_channels = input_dict["in_channels"]
        out_channels = input_dict["out_channels"]
        kernel_size = input_dict["kernel_size"]
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        padding_mode = input_dict.get("padding_mode", 'zeros')
        bias = input_dict.get("bias", True)

        mod = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, bias=bias)
        mod.weight = torch.nn.Parameter(torch.tensor(input_dict["weight"]))
        if bias:
            mod.bias = torch.nn.Parameter(torch.tensor(input_dict["bias_tensor"]))
        input_tensor = torch.tensor(input_dict["input"])

        if not cpu:
            mod = mod.cuda()
            input_tensor = input_tensor.cuda()

        result = mod(input_tensor)

        if not cpu:
            result = result.cpu()
        
        return {"result": result.numpy()}
    
    elif module_type == 'LinearReLU':
        in_features = input_dict["in_features"]
        out_features = input_dict["out_features"]
        bias = input_dict.get("bias", True)
        
        lin = nn.Linear(in_features, out_features, bias=bias)
        
        input_tensor = torch.tensor(input_dict["input"])
        mod = ConvReLU2d(lin)
        

        if not cpu:
            mod = mod.cuda()
            input_tensor = input_tensor.cuda()

        result = mod(input_tensor)

        if not cpu:
            result = result.cpu()
        
        return {"result": result.numpy()}

    elif module_type == 'Linear':
        in_features = input_dict["in_features"]
        out_features = input_dict["out_features"]
        bias = input_dict.get("bias", True)
        
        mod = nn.Linear(in_features, out_features, bias=bias)
        mod.weight = torch.nn.Parameter(torch.tensor(input_dict["weight"]))
        if bias:
            mod.bias = torch.nn.Parameter(torch.tensor(input_dict["bias_tensor"]))
        input_tensor = torch.tensor(input_dict["input"])

        if not cpu:
            mod = mod.cuda()
            input_tensor = input_tensor.cuda()

        result = mod(input_tensor)

        if not cpu:
            result = result.cpu()
        
        return {"result": result.numpy()}
    
    else:
        raise ValueError("Unsupported module type")

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    from tensorflow.keras.layers import Conv2D, Dense, ReLU
    from tensorflow.keras import Model
    
    module_type = input_dict['module_type']
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        if module_type == 'ConvReLU2d':
            in_channels = input_dict["in_channels"]
            out_channels = input_dict["out_channels"]
            kernel_size = input_dict["kernel_size"]
            stride = input_dict.get("stride", 1)
            padding = input_dict.get("padding", 0)
            dilation = input_dict.get("dilation", 1)
            groups = input_dict.get("groups", 1)
            padding_mode = input_dict.get("padding_mode", 'valid')
            bias = input_dict.get("bias", True)

            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)

            if isinstance(stride, int):
                stride = (stride, stride)

            if isinstance(padding, int):
                padding = (padding, padding)
                
            input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
            input_tensor = tf.reshape(input_tensor, [1, input_tensor.shape[0], input_tensor.shape[1], in_channels])
            
            weight = tf.constant(input_dict["weight"], dtype=tf.float32)
            weight = tf.transpose(weight, perm=[2, 3, 1, 0]) 

            if bias:
                bias_tensor = tf.constant(input_dict["bias_tensor"], dtype=tf.float32)

            conv = Conv2D(out_channels, kernel_size, strides=stride, padding='same', dilation_rate=dilation, use_bias=bias, groups=groups,
                          kernel_initializer=tf.constant_initializer(weight.numpy()),
                          bias_initializer=tf.constant_initializer(bias_tensor.numpy() if bias else None))
            
            relu = ReLU()

            result = conv(input_tensor)
            result = relu(result)
            result = result.numpy()
            result = result.reshape(result.shape[1:])
            
            return {"result": result}

        elif module_type == 'Conv2d':
            in_channels = input_dict["in_channels"]
            out_channels = input_dict["out_channels"]
            kernel_size = input_dict["kernel_size"]
            stride = input_dict.get("stride", 1)
            padding = input_dict.get("padding", 0)
            dilation = input_dict.get("dilation", 1)
            groups = input_dict.get("groups", 1)
            padding_mode = input_dict.get("padding_mode", 'valid')
            bias = input_dict.get("bias", True)

            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)

            if isinstance(stride, int):
                stride = (stride, stride)

            if isinstance(padding, int):
                padding = (padding, padding)
                
            input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
            input_tensor = tf.reshape(input_tensor, [1, input_tensor.shape[0], input_tensor.shape[1], in_channels])
            
            weight = tf.constant(input_dict["weight"], dtype=tf.float32)
            weight = tf.transpose(weight, perm=[2, 3, 1, 0])

            if bias:
                bias_tensor = tf.constant(input_dict["bias_tensor"], dtype=tf.float32)
                
            conv = Conv2D(out_channels, kernel_size, strides=stride, padding='same', dilation_rate=dilation, use_bias=bias, groups=groups,
                          kernel_initializer=tf.constant_initializer(weight.numpy()),
                          bias_initializer=tf.constant_initializer(bias_tensor.numpy() if bias else None))
            
            result = conv(input_tensor)
            result = result.numpy()
            result = result.reshape(result.shape[1:])
            
            return {"result": result}
        
        elif module_type == 'LinearReLU':
            in_features = input_dict["in_features"]
            out_features = input_dict["out_features"]
            bias = input_dict.get("bias", True)
            
            input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
            input_tensor = tf.reshape(input_tensor, [1, in_features])
            
            weight = tf.constant(input_dict["weight"], dtype=tf.float32)
            weight = tf.transpose(weight, perm=[1, 0])

            if bias:
                bias_tensor = tf.constant(input_dict["bias_tensor"], dtype=tf.float32)
            
            dense = Dense(out_features, use_bias=bias,
                          kernel_initializer=tf.constant_initializer(weight.numpy()),
                          bias_initializer=tf.constant_initializer(bias_tensor.numpy() if bias else None))
            relu = ReLU()

            result = dense(input_tensor)
            result = relu(result)
            result = result.numpy()
            result = result.reshape(result.shape[1:])
            
            return {"result": result}

        elif module_type == 'Linear':
            in_features = input_dict["in_features"]
            out_features = input_dict["out_features"]
            bias = input_dict.get("bias", True)
            
            input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
            input_tensor = tf.reshape(input_tensor, [1, in_features])
            
            weight = tf.constant(input_dict["weight"], dtype=tf.float32)
            weight = tf.transpose(weight, perm=[1, 0])

            if bias:
                bias_tensor = tf.constant(input_dict["bias_tensor"], dtype=tf.float32)
                
            dense = Dense(out_features, use_bias=bias,
                          kernel_initializer=tf.constant_initializer(weight.numpy()),
                          bias_initializer=tf.constant_initializer(bias_tensor.numpy() if bias else None))

            result = dense(input_tensor)
            result = result.numpy()
            result = result.reshape(result.shape[1:])
            
            return {"result": result}
        else:
            raise ValueError("Unsupported module type")

def main():
    A_TOL = 0.01

    input_data_convrelu = {
        "module_type": 'ConvReLU2d',
        "in_channels": 3,
        "out_channels": 2,
        "kernel_size": 3,
        "input": np.random.rand(10, 10, 3).astype(np.float32),
    }

    input_data_convrelu["weight"] = np.random.rand(input_data_convrelu["out_channels"], input_data_convrelu["in_channels"], input_data_convrelu["kernel_size"], input_data_convrelu["kernel_size"]).astype(np.float32)
    input_data_convrelu["bias_tensor"] = np.random.rand(input_data_convrelu["out_channels"]).astype(np.float32)


    torch_result_convrelu = torch_version(input_data_convrelu)
    tf_result_convrelu = tensorflow_version(input_data_convrelu)
    
    assert np.allclose(torch_result_convrelu["result"], tf_result_convrelu["result"], atol=A_TOL), "Results do not match for ConvReLU2d"

    input_data_conv = {
        "module_type": 'Conv2d',
        "in_channels": 3,
        "out_channels": 2,
        "kernel_size": 3,
        "input": np.random.rand(10, 10, 3).astype(np.float32),
    }
    input_data_conv["weight"] = np.random.rand(input_data_conv["out_channels"], input_data_conv["in_channels"], input_data_conv["kernel_size"], input_data_conv["kernel_size"]).astype(np.float32)
    input_data_conv["bias_tensor"] = np.random.rand(input_data_conv["out_channels"]).astype(np.float32)



    torch_result_conv = torch_version(input_data_conv)
    tf_result_conv = tensorflow_version(input_data_conv)
    
    assert np.allclose(torch_result_conv["result"], tf_result_conv["result"], atol=A_TOL), "Results do not match for Conv2d"

    input_data_linearrelu = {
        "module_type": 'LinearReLU',
        "in_features": 5,
        "out_features": 4,
        "input": np.random.rand(5).astype(np.float32),
    }
    input_data_linearrelu["weight"] = np.random.rand(input_data_linearrelu["out_features"], input_data_linearrelu["in_features"]).astype(np.float32)
    input_data_linearrelu["bias_tensor"] = np.random.rand(input_data_linearrelu["out_features"]).astype(np.float32)



    torch_result_linearrelu = torch_version(input_data_linearrelu)
    tf_result_linearrelu = tensorflow_version(input_data_linearrelu)
    
    assert np.allclose(torch_result_linearrelu["result"], tf_result_linearrelu["result"], atol=A_TOL), "Results do not match for LinearReLU"

    input_data_linear = {
        "module_type": 'Linear',
        "in_features": 5,
        "out_features": 4,
        "input": np.random.rand(5).astype(np.float32),
    }

    input_data_linear["weight"] = np.random.rand(input_data_linear["out_features"], input_data_linear["in_features"]).astype(np.float32)
    input_data_linear["bias_tensor"] = np.random.rand(input_data_linear["out_features"]).astype(np.float32)



    torch_result_linear = torch_version(input_data_linear)
    tf_result_linear = tensorflow_version(input_data_linear)
    
    assert np.allclose(torch_result_linear["result"], tf_result_linear["result"], atol=A_TOL), "Results do not match for Linear"

    print("Success")

if __name__ == "__main__":
    main()