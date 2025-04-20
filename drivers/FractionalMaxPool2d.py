import numpy as np

def torch_version(input, cpu=True):
    import torch
    """ PyTorch implementation of FractionalMaxPooling function """

    # Extract parameters from the input dictionary
    kernel_size = input['kernel_size']
    output_size = input.get('output_size', None)
    output_ratio = input.get('output_ratio', None)
    return_indices = input.get('return_indices', False)
    _random_samples = input.get('_random_samples', None)
    
    if output_ratio is not None:
        # Initialize FractionalMaxPool2d with given parameters
        pool = torch.nn.FractionalMaxPool2d(kernel_size, output_ratio=output_ratio, return_indices=return_indices, _random_samples=_random_samples)
    else:
        pool = torch.nn.FractionalMaxPool2d(kernel_size, output_size=output_size, return_indices=return_indices, _random_samples=_random_samples)
    
    # Convert input to PyTorch tensor
    input_tensor = torch.tensor(input['input'])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        pool = pool.cuda()
    
    # Apply pooling operation
    output = pool(input_tensor)

    return {"output": output.cpu().detach().numpy()} if not return_indices else {"output": output[0].cpu().detach().numpy(), "indices": output[1].cpu().detach().numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    """ TensorFlow approximation of FractionalMaxPooling function """

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Extract parameters from the input dictionary
        kernel_size = input['kernel_size']
        output_size = input.get('output_size', None)
        output_ratio = input.get('output_ratio', None)
        
        assert output_size is not None or output_ratio is not None
        
        input_tensor = tf.constant(input['input'])
        input_shape = tf.shape(input_tensor)

        # Compute the output size from the provided output ratio if necessary
        if output_size is None:
            output_size = (int(output_ratio[0] * input_shape[2]), int(output_ratio[1] * input_shape[3]))
        
        batch_size, num_channels, height, width = input_tensor.shape
        input_tensor = tf.transpose(input_tensor, [0, 2, 3, 1])  # TensorFlow expects NHWC format
        
        # Calculate strides and pooling window size based on output size
        output_height, output_width = output_size
        height_stride = height // output_height
        width_stride = width // output_width

        # Perform pooling operation using calculated parameters
        overlay_h = (height - height_stride * output_height + output_height - 1) // output_height
        overlay_w = (width - width_stride * output_width + output_width - 1) // output_width

        # Use sliding window to simulate behavior of fractional max pool
        pooled_output = tf.image.extract_patches(
            images=input_tensor,
            sizes=[1, height_stride + overlay_h, width_stride + overlay_w, 1],
            strides=[1, height_stride, width_stride, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        
        pooling_windows = tf.reshape(pooled_output, [batch_size, output_height, output_width, height_stride + overlay_h, width_stride + overlay_w, num_channels])
        pooling_windows = tf.reduce_max(pooling_windows, axis=[3, 4])

        output = tf.transpose(pooling_windows, [0, 3, 1, 2])  # Convert back to NCHW for comparison
        
        return {"output": output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(2, 3, 6, 6).astype(np.float32),  # Random input tensor
        "kernel_size": 2,
        "output_size": (3, 3),
        "output_ratio": None,
        "return_indices": False,
        "_random_samples": None
    }

    # PyTorch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert that results are equal
    if np.allclose(torch_result["output"], tf_result["output"], atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()