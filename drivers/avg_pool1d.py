import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"]).unsqueeze(1)  # Adding an input channel dimension for 1D pooling
    kernel_size = input["kernel_size"]
    stride = input.get("stride", None)
    padding = input.get("padding", 0)
    ceil_mode = input.get("ceil_mode", False)
    count_include_pad = input.get("count_include_pad", True)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    # Apply to torch.nn.functional.avg_pool1d
    output_tensor = torch.nn.functional.avg_pool1d(
        input_tensor, kernel_size, stride=stride, padding=padding, 
        ceil_mode=ceil_mode, count_include_pad=count_include_pad
    )

    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"avg_pool1d_output": output_tensor.squeeze(1).numpy()}  # Remove input channel dimension

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        input_tensor = tf.expand_dims(input_tensor, axis=-1)  # Adding an input channel dimension for 1D pooling
        kernel_size = input["kernel_size"]
        stride = input.get("stride", None)
        padding = input.get("padding", 0)
        ceil_mode = input.get("ceil_mode", False)
        count_include_pad = input.get("count_include_pad", True)

        # Determine padding
        if padding == 0:
            padding_str = 'VALID'
        else:
            padding_str = 'SAME'

        # Default value for stride
        if stride is None:
            stride = kernel_size

        # Apply to TensorFlow equivalent
        output_tensor = tf.nn.pool(
            input_tensor, window_shape=[kernel_size], pooling_type='AVG', padding=padding_str, strides=[stride],
            data_format='NWC', dilations=[1]
        )

        return {"avg_pool1d_output": tf.squeeze(output_tensor, axis=-1).numpy()}  # Remove input channel dimension

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8, 0.2, 0.6, 0.9]], dtype=np.float32),
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True
    }
    

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results with a tolerance
    tolerance = 1e-5
    if np.allclose(torch_result["avg_pool1d_output"], tf_result["avg_pool1d_output"], atol=tolerance):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()