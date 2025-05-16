import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    kernel_size = input_dict.get("kernel_size")
    stride = input_dict.get("stride", None)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    return_indices = input_dict.get("return_indices", False)
    ceil_mode = input_dict.get("ceil_mode", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)(input_tensor)

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
        kernel_size = input_dict.get("kernel_size")
        stride = input_dict.get("stride", None)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        return_indices = input_dict.get("return_indices", False)
        ceil_mode = input_dict.get("ceil_mode", False)
        
        if isinstance(padding, int):
            padding = [padding, padding, padding]

        if stride is None:
            stride = kernel_size

        if isinstance(stride, int):
            stride = [stride, stride, stride]

        if isinstance(dilation, int):
            dilation = [dilation, dilation, dilation]

        
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        
        padding_arg = 'VALID'
        if padding != [0,0,0]:
            padding_arg = 'SAME'
        
        result = tf.nn.max_pool3d(
            input=input_tensor,
            ksize=[1, kernel_size[0], kernel_size[1], kernel_size[2], 1],
            strides=[1, stride[0], stride[1], stride[2], 1],
            padding=padding_arg,
            data_format='NCDHW'
        )
        
        result = tf.squeeze(result, axis=0)
        result = result.numpy()
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 10, 10, 10).astype(np.float32),
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": (1, 1, 1),
    }

    torch_result = torch_version(input_data)
    
    # Reshape torch result to match the expected output shape from TensorFlow.
    torch_result["result"] = torch_result["result"].transpose(1, 2, 3, 0)

    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()