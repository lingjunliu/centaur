import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    
    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    size = input.get("size", None)
    scale_factor = input.get("scale_factor", None)
    mode = input.get("mode", 'nearest')
    align_corners = input.get("align_corners", None)
    
    # Ensure proper arguments for size and scale_factor
    if size is not None:
        size = tuple(size if isinstance(size, (list, tuple)) else (size,))
    if scale_factor is not None:
        scale_factor = tuple(scale_factor if isinstance(scale_factor, (list, tuple)) else (scale_factor,))
    
    # Apply to torch.nn.Upsample
    upsample = torch.nn.Upsample(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
    output = upsample(input_tensor.permute(0, 3, 1, 2))  # Change to channels-first for PyTorch

    if not cpu:
        output = output.cpu()

    return {"output": output.permute(0, 2, 3, 1).detach().numpy()}  # Change back to channels-last for comparison

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
        size = input.get("size", None)
        scale_factor = input.get("scale_factor", None)
        mode = input.get("mode", 'nearest')

        align_corners = input.get("align_corners", None)
        
        # Calculate new size if scale_factor is provided
        if scale_factor is not None:
            scale_factor = np.array(scale_factor if isinstance(scale_factor, (list, tuple)) else [scale_factor], dtype=np.float32)
            new_size = np.array(input_tensor.shape[1:3]) * scale_factor
            size = new_size.astype(int).tolist()

        # Ensure size is a tuple/list
        if size is not None:
            size = size if isinstance(size, (list, tuple)) else [size]
        
        # Apply to TensorFlow equivalent
        if mode == 'nearest':
            method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        elif mode == 'bilinear':
            method = tf.image.ResizeMethod.BILINEAR
        elif mode == 'bicubic':
            method = tf.image.ResizeMethod.BICUBIC
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        resized_image = tf.image.resize(input_tensor, size, method=method, preserve_aspect_ratio=False)

        return {"output": resized_image.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.rand(1, 2, 2, 2).astype(np.float32), # Ensuring it's 4D for the example
        "size": [4, 4], # or scale_factor, but not both
        "scale_factor": None,
        "mode": 'nearest',
        "align_corners": None
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare the results
    torch_output = torch_result["output"]
    tf_output = tf_result["output"]
    
    if np.allclose(torch_output, tf_output, atol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()