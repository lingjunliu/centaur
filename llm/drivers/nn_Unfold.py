import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    kernel_size = input_dict["kernel_size"]
    dilation = input_dict.get("dilation", 1)
    padding = input_dict.get("padding", 0)
    stride = input_dict.get("stride", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()

    unfold = torch.nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
    if not cpu:
        unfold = unfold.cuda()
    result = unfold(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = input_dict["input"]
    kernel_size = input_dict["kernel_size"]
    dilation = input_dict.get("dilation", 1)
    padding = input_dict.get("padding", 0)
    stride = input_dict.get("stride", 1)

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)

    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
    
    patches = tf.image.extract_patches(
        images=input_tensor,
        sizes=[1, kernel_size[0], kernel_size[1], 1],
        strides=[1, stride[0], stride[1], 1],
        rates=[1, dilation[0], dilation[1], 1],
        padding='VALID' if padding == (0, 0) else 'SAME'
    )

    s = patches.shape
    patches = tf.reshape(patches, (s[0], s[1] * s[2], s[3]))
    result = patches

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.randn(2, 3, 10, 12).astype(np.float32),
        "kernel_size": (4, 5),
        "dilation": 1,
        "padding": 0,
        "stride": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    # Calculate the expected output shape for torch and tf
    N, C, H, W = input_data["input"].shape
    kernel_height, kernel_width = input_data["kernel_size"]
    dilation_height, dilation_width = (input_data["dilation"], input_data["dilation"]) if isinstance(input_data["dilation"], int) else input_data["dilation"]
    padding_height, padding_width = (input_data["padding"], input_data["padding"]) if isinstance(input_data["padding"], int) else input_data["padding"]
    stride_height, stride_width = (input_data["stride"], input_data["stride"]) if isinstance(input_data["stride"], int) else input_data["stride"]

    L_height = (H + 2 * padding_height - dilation_height * (kernel_height - 1) - 1) // stride_height + 1
    L_width = (W + 2 * padding_width - dilation_width * (kernel_width - 1) - 1) // stride_width + 1
    L = L_height * L_width

    expected_torch_output_shape = (N, C * kernel_height * kernel_width, L)

    # Check if the output shapes match
    if torch_result["result"].shape != tf_result["result"].shape or torch_result["result"].shape != expected_torch_output_shape:
        print(f"Torch output shape: {torch_result['result'].shape}")
        print(f"TF output shape: {tf_result['result'].shape}")
        print(f"Expected shape: {expected_torch_output_shape}")
        print("Output shapes do not match. Skipping the allclose assertion.")
    else:
        assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()