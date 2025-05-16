import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    
    # Unpack input dictionary
    input_tensor = torch.tensor(input_dict["input"])
    
    # Create MaxPool2d layer
    maxpool = torch.nn.MaxPool2d(
        kernel_size=input_dict["kernel_size"],
        stride=input_dict.get("stride", input_dict["kernel_size"]),
        padding=input_dict.get("padding", 0),
        dilation=input_dict.get("dilation", 1),
        return_indices=input_dict.get("return_indices", False),
        ceil_mode=input_dict.get("ceil_mode", False)
    )
    
    # Apply pooling
    with torch.no_grad():
        if not cpu and torch.cuda.is_available():
            maxpool = maxpool.cuda()
            input_tensor = input_tensor.cuda()
        
        output = maxpool(input_tensor)
    
    return output.cpu().numpy() if isinstance(output, torch.Tensor) else (output[0].cpu().numpy(), output[1].cpu().numpy())

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    # Unpack input dictionary
    input_tensor = tf.constant(input_dict["input"])

    strides = input_dict.get("stride", input_dict["kernel_size"])
    if isinstance(strides, int):
        strides = [1, strides, strides, 1]
    else:
        strides = [1, strides[0], strides[1], 1]

    ksize = input_dict["kernel_size"]
    if isinstance(ksize, int):
        ksize = [1, ksize, ksize, 1]
    else:
        ksize = [1, ksize[0], ksize[1], 1]

    padding = "SAME" if input_dict.get("padding", 0) > 0 else "VALID"
    
    with tf.device(device_string):
        output = tf.nn.max_pool2d(
            input_tensor,
            ksize=ksize,
            strides=strides,
            padding=padding
        )
    
    return output.numpy()

def main():
    # Example input
    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version({
        "input": np.transpose(input_data["input"], (0, 2, 3, 1)),  # Convert to NHWC format for TensorFlow
        "kernel_size": input_data["kernel_size"],
        "stride": input_data["stride"],
        "padding": input_data["padding"],
        "dilation": input_data["dilation"],
        "return_indices": input_data["return_indices"],
        "ceil_mode": input_data["ceil_mode"]
    })
    print("TensorFlow result:", np.transpose(tf_result, (0, 3, 1, 2)))  # Convert back to NCHW format for comparison

    # Convert both results to numpy arrays for comparison
    torch_result_np = torch_result
    tf_result_np = np.transpose(tf_result, (0, 3, 1, 2))

    if np.allclose(torch_result_np, tf_result_np, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()