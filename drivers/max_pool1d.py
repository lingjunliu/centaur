import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    
    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    kernel_size = input["kernel_size"]
    stride = input.get("stride", kernel_size)  # Default to kernel_size if not provided
    padding = input.get("padding", 0)
    dilation = input.get("dilation", 1)
    ceil_mode = input.get("ceil_mode", False)
    
    # Apply to torch.nn.functional.max_pool1d
    output = torch.nn.functional.max_pool1d(
        input_tensor, kernel_size, stride, padding, dilation, ceil_mode
    )
    
    if not cpu:
        output = output.cpu()
    
    return {"max_pool1d_output": output.detach().numpy()}

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
        kernel_size = input["kernel_size"]
        stride = input.get("stride", kernel_size)  # Default to kernel_size if not provided
        padding = input.get("padding", 0)
        dilation = input.get("dilation", 1)
        ceil_mode = input.get("ceil_mode", False)
        
        # TensorFlow does not support dilation in max pooling.
        # We will adjust the algorithm to fit TensorFlow's max pooling
        if dilation != 1:
            raise NotImplementedError("Dilation in max pooling is not supported in TensorFlow")
        
        # Adjust padding
        padding_mode = 'SAME' if padding != 0 else 'VALID'
        
        # Reshape input to the required format (N, C, W) for 1D pooling
        input_tensor_4d = tf.reshape(input_tensor, [input_tensor.shape[0], input_tensor.shape[2], 1, input_tensor.shape[1]])
        
        # Apply to TensorFlow equivalent
        output = tf.nn.max_pool(
            input_tensor_4d,
            ksize=[1, kernel_size, 1, 1],
            strides=[1, stride, 1, 1],
            padding=padding_mode
        )
        
        output = tf.reshape(output, [output.shape[0], output.shape[3], output.shape[1]])
        
        return {"max_pool1d_output": output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8, 0.9], [0.2, 0.6, 0.9, 0.4]], dtype=np.float32).reshape((2, 1, 4)),
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "ceil_mode": False
    }
    
    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)
    
    # Assertion to check equality
    torch_output = torch_result["max_pool1d_output"]
    tf_output = tf_result["max_pool1d_output"]
    
    if np.allclose(torch_output, tf_output, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()