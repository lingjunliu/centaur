import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    output_size = input_dict["output_size"]
    kernel_size = input_dict["kernel_size"]
    dilation = input_dict.get("dilation", 1)
    padding = input_dict.get("padding", 0)
    stride = input_dict.get("stride", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()

    fold = torch.nn.Fold(output_size=output_size, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
    result = fold(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    output_size = input_dict["output_size"]
    kernel_size = input_dict["kernel_size"]
    dilation = input_dict.get("dilation", 1)
    padding = input_dict.get("padding", 0)
    stride = input_dict.get("stride", 1)

    if isinstance(output_size, int):
      output_size = (output_size, output_size)
    if isinstance(kernel_size, int):
      kernel_size = (kernel_size, kernel_size)
    if isinstance(dilation, int):
      dilation = (dilation, dilation)
    if isinstance(padding, int):
      padding = (padding, padding)
    if isinstance(stride, int):
      stride = (stride, stride)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        n = 1 if len(input_tensor.shape) == 2 else input_tensor.shape[0]
        c_k_prod = input_tensor.shape[0] if len(input_tensor.shape) == 2 else input_tensor.shape[1]
        l = input_tensor.shape[1] if len(input_tensor.shape) == 2 else input_tensor.shape[2]

        channels = c_k_prod // (kernel_size[0] * kernel_size[1])

        output_height, output_width = output_size

        output = np.zeros((n, channels, output_height, output_width), dtype=np.float32)
        
        num_blocks_h = (output_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
        num_blocks_w = (output_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1

        input_np = input_tensor.numpy()
        if len(input_np.shape) == 2:
            input_np = input_np[None, :, :]

        for b in range(n):
            for i in range(num_blocks_h):
                for j in range(num_blocks_w):
                    start_h = i * stride[0] - padding[0]
                    start_w = j * stride[1] - padding[1]

                    for ky in range(kernel_size[0]):
                        for kx in range(kernel_size[1]):
                            h = start_h + dilation[0] * ky
                            w = start_w + dilation[1] * kx

                            if 0 <= h < output_height and 0 <= w < output_width:
                                block_idx = i * num_blocks_w + j
                                output[b, :, h, w] += input_np[b, :(channels * kernel_size[0] * kernel_size[1]), block_idx].reshape((channels, kernel_size[0] * kernel_size[1]))[:, ky * kernel_size[1] + kx]

        result = output

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]], dtype=np.float32),
        "output_size": (4, 5),
        "kernel_size": (2, 2)
    }
    input_data["input"] = np.random.rand(1, 3 * 2 * 2, 12).astype(np.float32)

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()