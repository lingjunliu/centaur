import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.functional.fold(input_tensor, output_size=input_dict["output_size"], kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    
    input_tensor = tf.constant(input_dict["input"])
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    output_size = input_dict["output_size"]
    
    input_shape = input_tensor.shape
    
    if isinstance(kernel_size, int):
      kernel_size = [kernel_size, kernel_size]
    
    if isinstance(output_size, int):
      output_size = [output_size, output_size]

    if isinstance(stride, int):
        stride = [stride, stride]

    if isinstance(padding, int):
        padding = [padding, padding]

    if isinstance(dilation, int):
        dilation = [dilation, dilation]

    input_tensor_np = input_tensor.numpy()

    unfolded_height, unfolded_width = output_size
    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    
    input_channels = input_shape[-1]
    
    output_height = (unfolded_height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    output_width = (unfolded_width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    
    output_tensor = np.zeros((1, input_channels, unfolded_height, unfolded_width), dtype=input_tensor_np.dtype)
    
    input_tensor_np = np.reshape(input_tensor_np, (input_channels, output_height, output_width))
    
    for i in range(output_height):
        for j in range(output_width):
            row_start = i * sh - ph
            row_end = row_start + dh * (kh - 1) + 1
            col_start = j * sw - pw
            col_end = col_start + dw * (kw - 1) + 1
            
            row_slice_start = max(0, row_start)
            row_slice_end = min(unfolded_height, row_end)
            col_slice_start = max(0, col_start)
            col_slice_end = min(unfolded_width, col_end)
            
            kernel_row_start = (row_slice_start - row_start) // dh
            kernel_row_end = kernel_row_start + (row_slice_end - row_slice_start + dh - 1) // dh
            kernel_col_start = (col_slice_start - col_start) // dw
            kernel_col_end = kernel_col_start + (col_slice_end - col_slice_start + dw - 1) // dw
            
            output_tensor[0, :, row_slice_start:row_slice_end, col_slice_start:col_slice_end] += \
                input_tensor_np[:, i, j][:, None, None]
    
    return {"result": output_tensor}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3 * 5 * 5).astype(np.float32),
        "output_size": (10, 10),
        "kernel_size": (5, 5),
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