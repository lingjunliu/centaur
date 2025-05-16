import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None
    padding = input_dict.get("padding", 0)
    stride = input_dict.get("stride", 1)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)

    if not isinstance(padding, tuple):
        padding = (padding, padding)
    if not isinstance(stride, tuple):
        stride = (stride, stride)
    if not isinstance(dilation, tuple):
        dilation = (dilation, dilation)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()
    else:
        raise NotImplementedError("torch.cudnn_convolution_relu only supports CUDA backend")

    result = torch.cudnn_convolution_relu(input_tensor, weight, bias, padding, stride, dilation, groups)

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
        bias = tf.constant(input_dict["bias"]) if "bias" in input_dict else None
        padding = input_dict.get("padding", 0)
        stride = input_dict.get("stride", 1)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)

        if not isinstance(padding, tuple):
            padding = (padding, padding)
        if not isinstance(stride, tuple):
            stride = (stride, stride)
        if not isinstance(dilation, tuple):
            dilation = (dilation, dilation)
            
        padding_list = [(0, 0), (padding[0], padding[0]), (padding[1], padding[1]), (0, 0)] 

        input_reshaped = tf.reshape(input_tensor, [1, input_tensor.shape[0], input_tensor.shape[1], 1])
        weight_reshaped = tf.reshape(weight, [weight.shape[0], weight.shape[1], 1, 1])

        if padding != (0,0):
            input_padded = tf.pad(input_reshaped, padding_list, "CONSTANT")
        else:
            input_padded = input_reshaped

        strides_list = [1, stride[0], stride[1], 1] 
        dilations_list = [1, dilation[0], dilation[1], 1]

        conv_out = tf.nn.conv2d(input_padded, weight_reshaped, strides=strides_list, padding="VALID", dilations=dilations_list)
        
        if bias is not None:
            conv_out = tf.nn.bias_add(conv_out, tf.reshape(bias, [-1]))
        
        relu_out = tf.nn.relu(conv_out)

        result = tf.reshape(relu_out, [relu_out.shape[1], relu_out.shape[2]]).numpy()
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "weight": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "bias": np.array([0.5], dtype=np.float32),
        "padding": 0,
        "stride": 1,
        "dilation": 1,
        "groups": 1
    }

    try:
        torch_result = torch_version(input_data, cpu=False)
        tf_result = tensorflow_version(input_data, cpu=False)

        assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

        print("Success")
    except NotImplementedError as e:
        print(f"Skipping test due to: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()