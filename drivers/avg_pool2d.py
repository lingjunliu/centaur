import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    kernel_size = input.get("kernel_size", 2)
    stride = input.get("stride", None)
    padding = input.get("padding", 0)
    ceil_mode = input.get("ceil_mode", False)
    count_include_pad = input.get("count_include_pad", True)
    divisor_override = input.get("divisor_override", None)
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply torch.avg_pool2d
    result = torch.nn.functional.avg_pool2d(
        input_tensor, kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, 
        count_include_pad=count_include_pad, divisor_override=divisor_override
    )

    if not cpu:
        result = result.cpu()

    return {"avg_pool2d_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
        
    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        ksize = input.get("kernel_size", 2)
        strides = input.get("stride", ksize)
        padding = 'VALID' if input.get("padding", 0) == 0 else 'SAME'
        
        # TensorFlow equivalent does not support all the parameters directly like PyTorch
        
        result = tf.nn.avg_pool(
            input_tensor, ksize=[1, ksize, ksize, 1], strides=[1, strides, strides, 1], padding=padding
        )

        return {"avg_pool2d_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=np.float32),
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    if np.allclose(torch_result["avg_pool2d_result"], tf_result["avg_pool2d_result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()