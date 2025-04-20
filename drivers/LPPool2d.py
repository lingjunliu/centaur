import numpy as np

def torch_version(input, cpu=True):
    import torch
    # Set the seed for reproducibility

    # Unpack input dictionary
    norm_type = input["norm_type"]
    kernel_size = input["kernel_size"]
    stride = input.get("stride", kernel_size)  # Default stride is equal to kernel size
    ceil_mode = input.get("ceil_mode", False)
    input_tensor = torch.tensor(input["input"])

    # Apply torch.nn.LPPool2d
    m = torch.nn.LPPool2d(norm_type, kernel_size, stride=stride, ceil_mode=ceil_mode)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        m = m.cuda()

    output_tensor = m(input_tensor)

    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"output": output_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    # Set the seed for reproducibility

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        norm_type = input["norm_type"]
        kernel_size = input["kernel_size"]
        stride = input.get("stride", kernel_size)  # Default stride is equal to kernel size
        ceil_mode = input.get("ceil_mode", False)
        input_tensor = tf.constant(input["input"])

        # Calculate the padding
        if ceil_mode:
            pad_total_height = max(kernel_size[0] - stride[0], 0)
            pad_total_width = max(kernel_size[1] - stride[1], 0)
            padding = [[0, 0], [0, 0], [pad_total_height, pad_total_height], [pad_total_width, pad_total_width]]
            input_tensor = tf.pad(input_tensor, paddings=padding)

        # Custom LPPool2D logic using TensorFlow operations
        def lp_pool_2d(tensor, norm_type, ksize, strides):
            batch, channels, height, width = tensor.shape
            k_h, k_w = ksize
            s_h, s_w = strides

            out_height = (height - k_h) // s_h + 1
            out_width = (width - k_w) // s_w + 1

            out = []
            for i in range(0, height - k_h + 1, s_h):
                row = []
                for j in range(0, width - k_w + 1, s_w):
                    patch = tensor[:, :, i:i + k_h, j:j + k_w]
                    if norm_type == np.inf:
                        value = tf.reduce_max(patch, axis=[-2, -1])
                    else:
                        value = tf.pow(tf.reduce_sum(tf.pow(patch, norm_type), axis=[-2, -1]), 1.0 / norm_type)
                    row.append(value)
                out.append(tf.stack(row, axis=-1))
            output_tensor = tf.stack(out, axis=-2)
            return output_tensor

        output_tensor = lp_pool_2d(input_tensor, norm_type, kernel_size, stride)
        
        return {"output": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "norm_type": 2,
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "ceil_mode": False,
        "input": np.random.randn(1, 3, 32, 32).astype(np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_output = torch_result["output"]
    tf_output = tf_result["output"]

    # Ensuring the outputs are close enough
    if np.allclose(torch_output, tf_output, atol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()