import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from torch.nn.intrinsic.qat.modules import ConvBn1d, ConvBn2d, ConvBn3d, ConvBnReLU1d, ConvBnReLU2d, ConvBnReLU3d
    from torch.quantization import QConfig, get_default_qconfig
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn

    for key in input_dict:
        if isinstance(input_dict[key], np.ndarray):
            input_dict[key] = torch.tensor(input_dict[key])
    
    if not cpu:
        for key in input_dict:
            input_dict[key] = input_dict[key].cuda()
    
    if "module" in input_dict:
        module = input_dict["module"]
        del input_dict["module"]
    else:
        module = None

    qconfig = get_default_qconfig("fbgemm")
        
    if module == "ConvBn1d":
        result = ConvBn1d(qconfig=qconfig, **input_dict)
    elif module == "ConvBn2d":
        result = ConvBn2d(qconfig=qconfig, **input_dict)
    elif module == "ConvBn3d":
        result = ConvBn3d(qconfig=qconfig, **input_dict)
    elif module == "ConvBnReLU1d":
        result = ConvBnReLU1d(qconfig=qconfig, **input_dict)
    elif module == "ConvBnReLU2d":
        result = ConvBnReLU2d(qconfig=qconfig, **input_dict)
    elif module == "ConvBnReLU3d":
        result = ConvBnReLU3d(qconfig=qconfig, **input_dict)
    elif module == "Linear":
        result = nn.Linear(input_dict["in_features"], input_dict["out_features"], bias=input_dict["bias"])
        for param in result.parameters():
            if not cpu:
                param.data = torch.randn(param.size()).cuda()
            else:
                param.data = torch.randn(param.size())
    else:
        raise ValueError("Unknown module")

    if module != "Linear":
        if not cpu:
            result = result.cpu()
        return {"result": result.weight.detach().numpy()}
    else:
        if not cpu:
            result = result.cpu()
        return {"result": list(result.parameters())[0].detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D, Dense, BatchNormalization, ReLU
    from tensorflow.keras.models import Sequential

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        for key in input_dict:
            if isinstance(input_dict[key], np.ndarray):
                input_dict[key] = tf.constant(input_dict[key])
        
        if "module" in input_dict:
            module = input_dict["module"]
            del input_dict["module"]
        else:
            module = None
            
        if module == "ConvBn1d":
            model = Sequential([
                Conv1D(filters=input_dict['out_channels'], kernel_size=input_dict['kernel_size'], strides=input_dict['stride'], padding='valid', use_bias=False, input_shape=(1, input_dict['in_channels'])),
                BatchNormalization()
            ])
            result = model.get_weights()[0]
        elif module == "ConvBn2d":
            model = Sequential([
                Conv2D(filters=input_dict['out_channels'], kernel_size=input_dict['kernel_size'], strides=input_dict['stride'], padding='valid', use_bias=False, input_shape=(1, 1, input_dict['in_channels'])),
                BatchNormalization()
            ])
            result = model.get_weights()[0]
        elif module == "ConvBn3d":
            model = Sequential([
                Conv3D(filters=input_dict['out_channels'], kernel_size=input_dict['kernel_size'], strides=input_dict['stride'], padding='valid', use_bias=False, input_shape=(1, 1, 1, input_dict['in_channels'])),
                BatchNormalization()
            ])
            result = model.get_weights()[0]
        elif module == "ConvBnReLU1d":
            model = Sequential([
                Conv1D(filters=input_dict['out_channels'], kernel_size=input_dict['kernel_size'], strides=input_dict['stride'], padding='valid', use_bias=False, input_shape=(1, input_dict['in_channels'])),
                BatchNormalization(),
                ReLU()
            ])
            result = model.get_weights()[0]
        elif module == "ConvBnReLU2d":
            model = Sequential([
                Conv2D(filters=input_dict['out_channels'], kernel_size=input_dict['kernel_size'], strides=input_dict['stride'], padding='valid', use_bias=False, input_shape=(1, 1, input_dict['in_channels'])),
                BatchNormalization(),
                ReLU()
            ])
            result = model.get_weights()[0]
        elif module == "ConvBnReLU3d":
            model = Sequential([
                Conv3D(filters=input_dict['out_channels'], kernel_size=input_dict['kernel_size'], strides=input_dict['stride'], padding='valid', use_bias=False, input_shape=(1, 1, 1, input_dict['in_channels'])),
                BatchNormalization(),
                ReLU()
            ])
            result = model.get_weights()[0]
        elif module == "Linear":
            model = Sequential([
                Dense(units=input_dict['out_features'], use_bias=False, input_shape=(input_dict['in_features'],))
            ])
            result = model.get_weights()[0]
        else:
            raise ValueError("Unknown module")

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data_convbn1d = {
        "module": "ConvBn1d",
        "in_channels": 3,
        "out_channels": 4,
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "bias": False,
        "padding_mode": 'zeros'
    }

    input_data_convbn2d = {
        "module": "ConvBn2d",
        "in_channels": 3,
        "out_channels": 4,
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "bias": False,
        "padding_mode": 'zeros'
    }
    
    input_data_convbn3d = {
        "module": "ConvBn3d",
        "in_channels": 3,
        "out_channels": 4,
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "bias": False,
        "padding_mode": 'zeros'
    }
    
    input_data_convbnrelu1d = {
        "module": "ConvBnReLU1d",
        "in_channels": 3,
        "out_channels": 4,
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "bias": False,
        "padding_mode": 'zeros'
    }
    
    input_data_convbnrelu2d = {
        "module": "ConvBnReLU2d",
        "in_channels": 3,
        "out_channels": 4,
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "bias": False,
        "padding_mode": 'zeros'
    }
    
    input_data_convbnrelu3d = {
        "module": "ConvBnReLU3d",
        "in_channels": 3,
        "out_channels": 4,
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "bias": False,
        "padding_mode": 'zeros'
    }
    
    input_data_linear = {
        "module": "Linear",
        "in_features": 3,
        "out_features": 4,
        "bias": False,
    }

    input_datas = [input_data_convbn1d, input_data_convbn2d, input_data_convbn3d, input_data_convbnrelu1d, input_data_convbnrelu2d, input_data_convbnrelu3d, input_data_linear]
    
    for input_data in input_datas:
        torch_result = torch_version(input_data)
        tf_result = tensorflow_version(input_data)
        
        assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()