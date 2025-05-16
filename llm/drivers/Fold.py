import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    output_size = input_dict["output_size"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    input_tensor = input_tensor.reshape(1, input_tensor.size(1) * input_tensor.size(2) * input_tensor.size(3), 1)

    result = torch.nn.functional.fold(input_tensor, output_size=output_size, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    output_size = input_dict["output_size"]

    input_shape = input_tensor.shape
    batch_size = input_shape[0]
    channels = input_shape[1]
    
    if isinstance(kernel_size, tuple):
        kernel_height, kernel_width = kernel_size
    else:
        kernel_height = kernel_size
        kernel_width = kernel_size
        
    if isinstance(stride, int):
        stride_height = stride_width = stride
    else:
        stride_height, stride_width = stride

    if isinstance(padding, int):
        padding_height = padding_width = padding
    else:
        padding_height, padding_width = padding

    if isinstance(dilation, int):
        dilation_height = dilation_width = dilation
    else:
        dilation_height, dilation_width = dilation
        
    output_height, output_width = output_size

    patches_height = kernel_height
    patches_width = kernel_width
    
    patches_height += (patches_height - 1) * (dilation_height - 1)
    patches_width += (patches_width - 1) * (dilation_width - 1)
    
    input_height = (output_height + 2 * padding_height - patches_height) // stride_height + 1
    input_width = (output_width + 2 * padding_width - patches_width) // stride_width + 1

    unfolded_dim = input_tensor.shape[-1]
    
    patches_tensor = tf.reshape(input_tensor, (batch_size, channels, 1, -1))

    patches_tensor = tf.transpose(patches_tensor, perm=[0, 3, 2, 1])

    patches_tensor = tf.reshape(patches_tensor, (batch_size * 1, patches_height, patches_width, channels))

    output_tensor = tf.nn.depthwise_conv2d(
        input=tf.zeros((batch_size * 1, output_height, output_width, channels), dtype=tf.float32),
        filter=patches_tensor,
        strides=[1, 1, 1, 1],
        padding='VALID',
        dilations=[1, 1, 1, 1]
    )

    output_tensor = tf.reshape(output_tensor, (batch_size, 1, output_height * output_width * channels))
    
    output_tensor = tf.reduce_sum(output_tensor, axis=1)
    output_tensor = tf.reshape(output_tensor, (batch_size, output_height, output_width, channels))
    output_tensor = tf.transpose(output_tensor, perm=[0, 3, 1, 2])
    

    return {"result": output_tensor.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[[1, 2, 3, 4, 5, 6, 7, 8, 9]]]], dtype=np.float32),
        "output_size": (3, 3),
        "kernel_size": (3, 3),
        "stride": 1,
        "padding": 0,
        "dilation": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()