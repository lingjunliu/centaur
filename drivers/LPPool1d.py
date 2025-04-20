import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    norm_type = input["norm_type"]
    kernel_size = input["kernel_size"]
    stride = input.get("stride", kernel_size)
    ceil_mode = input.get("ceil_mode", False)

    # Apply torch.nn.LPPool1d
    pool = torch.nn.LPPool1d(norm_type, kernel_size, stride, ceil_mode)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        pool = pool.cuda()

    output = pool(input_tensor)

    if not cpu:
        output = output.cpu()

    return {"lppool1d_output": output.detach().numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # TensorFlow doesn't have a direct equivalent, so we need to use custom implementation
    def lppool1d(tensor, norm_type, kernel_size, stride, ceil_mode):
        input_shape = tensor.shape
        batch_size, channels, length = input_shape

        pooled_length = ((length - kernel_size) // stride + 1) if not ceil_mode else int(np.ceil((length - kernel_size + 1) / stride))
        pooled_output = np.zeros((batch_size, channels, pooled_length))

        for b in range(batch_size):
            for c in range(channels):
                for i in range(pooled_length):
                    start = i * stride
                    end = start + kernel_size
                    window = tensor[b, c, start:end]

                    if norm_type == float('inf'):
                        pooled_output[b, c, i] = np.max(window)
                    else:
                        pooled_output[b, c, i] = np.power(np.sum(np.power(window, norm_type)), 1/norm_type)

        return pooled_output

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = input["input"]
        norm_type = input["norm_type"]
        kernel_size = input["kernel_size"]
        stride = input.get("stride", kernel_size)
        ceil_mode = input.get("ceil_mode", False)

        # Apply the custom LPPool1d
        output = lppool1d(input_tensor, norm_type, kernel_size, stride, ceil_mode)

        return {"lppool1d_output": output}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(20, 16, 50).astype(np.float32),
        "norm_type": 2,
        "kernel_size": 3,
        "stride": 2,
        "ceil_mode": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    torch_output = torch_result["lppool1d_output"]
    tf_output = tf_result["lppool1d_output"]

    if np.allclose(torch_output, tf_output, rtol=1e-3, atol=1e-3):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()