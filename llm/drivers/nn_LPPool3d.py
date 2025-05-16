import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    norm_type = input_dict["norm_type"]
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", None)
    ceil_mode = input_dict.get("ceil_mode", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    m = nn.LPPool3d(norm_type, kernel_size, stride=stride, ceil_mode=ceil_mode)

    if not cpu:
        m = m.cuda()
        
    result = m(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    norm_type = input_dict["norm_type"]
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", kernel_size)
    ceil_mode = input_dict.get("ceil_mode", False)

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)

    if isinstance(stride, int):
        stride = (stride, stride, stride)

    if not cpu:
        device_string = "/gpu:0"
    else:
        device_string = "/cpu:0"
    
    with tf.device(device_string):
        if norm_type == float('inf'):
            result = tf.nn.max_pool3d(
                input_tensor,
                ksize=(1, kernel_size[0], kernel_size[1], kernel_size[2], 1),
                strides=(1, stride[0], stride[1], stride[2], 1),
                padding='VALID' if not ceil_mode else 'VALID',
            )
        elif norm_type == 1:
            result = tf.nn.avg_pool3d(
                input_tensor,
                ksize=(1, kernel_size[0], kernel_size[1], kernel_size[2], 1),
                strides=(1, stride[0], stride[1], stride[2], 1),
                padding='VALID' if not ceil_mode else 'VALID',
            )
        else:
            input_power = tf.pow(tf.abs(input_tensor), norm_type)
            pooled_power = tf.nn.avg_pool3d(
                input_power,
                ksize=(1, kernel_size[0], kernel_size[1], kernel_size[2], 1),
                strides=(1, stride[0], stride[1], stride[2], 1),
                padding='VALID' if not ceil_mode else 'VALID',
            )
            result = tf.pow(pooled_power, 1.0 / norm_type)
        
        result = result.numpy()
    
    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 4, 5).astype(np.float32),
        "norm_type": 2,
        "kernel_size": (2, 2, 2),
        "stride": (1, 1, 1),
        "ceil_mode": False
    }

    torch_result = torch_version(input_data)
    
    tf_input = input_data['input']
    tf_input = np.transpose(tf_input, (1, 2, 3, 0))
    tf_input = np.expand_dims(tf_input, axis=0)

    input_data['input'] = tf_input
    tf_result = tensorflow_version(input_data)

    torch_result_np = torch_result['result']
    tf_result_np = tf_result['result'][0]
    tf_result_np = np.transpose(tf_result_np, (3, 0, 1, 2))
    
    assert np.allclose(torch_result_np, tf_result_np, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()