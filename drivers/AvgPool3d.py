import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    kernel_size = input.get("kernel_size")
    stride = input.get("stride", kernel_size)
    padding = input.get("padding", 0)
    ceil_mode = input.get("ceil_mode", False)
    count_include_pad = input.get("count_include_pad", True)
    divisor_override = input.get("divisor_override", None)

    # Apply to torch.nn.AvgPool3d
    pool = torch.nn.AvgPool3d(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override
    )
    
    output = pool(input_tensor)
    
    if not cpu:
        output = output.cpu()
    
    return {"output": output.numpy()}

def tensorflow_version(input, cpu=True):
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input["input"])
        kernel_size = input.get("kernel_size")
        stride = input.get("stride", kernel_size)
        padding = input.get("padding", 0)
        ceil_mode = input.get("ceil_mode", False)
        count_include_pad = input.get("count_include_pad", True)

        input_tensor = tf.transpose(input_tensor, [0, 2, 3, 4, 1])  # Convert NCDHW to NDHWC
        
        def average_pool3d(x):
            if isinstance(kernel_size, int):
                ksizes = [1, kernel_size, kernel_size, kernel_size, 1]
            else:
                ksizes = [1] + list(kernel_size) + [1]
            
            if isinstance(stride, int):
                strides = [1, stride, stride, stride, 1]
            else:
                strides = [1] + list(stride) + [1]

            paddings = "SAME" if padding > 0 else "VALID"

            x = tf.nn.avg_pool3d(
                x, ksize=ksizes, strides=strides, padding=paddings, data_format="NDHWC"
            )
            
            if ceil_mode:
                input_shape = input_tensor.shape
                output_shape = x.shape
                for i in range(1, 4):  # D, H, W
                    if (output_shape[i] - 1) * stride + 1 >= input_shape[i] + padding:
                        output_shape = list(output_shape)
                        output_shape[i] -= 1
                        x = tf.slice(x, [0] * len(output_shape), output_shape)
                
            if not count_include_pad:
                mask = tf.ones_like(x)
                mask = tf.nn.avg_pool3d(
                    mask, ksize=ksizes, strides=strides, padding=paddings, data_format="NDHWC"
                )
                mask = tf.where(mask == 0, 1.0, mask)
                x = x * mask
                
            return x
        
        output = average_pool3d(input_tensor)
        output = tf.transpose(output, [0, 4, 1, 2, 3])  # Convert back NDHWC to NCDHW

        return {"output": output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(1, 3, 8, 8, 8).astype(np.float32),
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result["output"][0])

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result["output"][0])

    # Comparison
    if np.allclose(torch_result["output"], tf_result["output"], atol=1e-6):
        print("Equal")
    else:
        print("Not equal")

if __name__ == "__main__":
    main()