import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    size = input.get("size", None)
    scale_factor = input.get("scale_factor", None)
    mode = input.get("mode", 'nearest')
    align_corners = input.get("align_corners", None)
    recompute_scale_factor = input.get("recompute_scale_factor", None)
    antialias = input.get("antialias", False)
    
    # Ensure the input tensor is in (N, C, H, W) format for spatial dimensions
    if len(input_tensor.shape) == 2:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Apply to torch.nn.functional.interpolate
    output_tensor = torch.nn.functional.interpolate(
        input=input_tensor, size=size, scale_factor=scale_factor, 
        mode=mode, align_corners=align_corners, 
        recompute_scale_factor=recompute_scale_factor, antialias=antialias
    )

    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"interpolated_tensor": output_tensor.squeeze(0).squeeze(0).numpy()}  # Remove added dimensions

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

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
        antialias = input.get("antialias", False)

        if scale_factor is not None:
            size = [int(dim * scale) for dim, scale in zip(input_tensor.shape, scale_factor)]
        
        # Ensure the input tensor is in (N, H, W, C) format for spatial dimensions
        if len(input_tensor.shape) == 2:
            input_tensor = tf.expand_dims(tf.expand_dims(input_tensor, 0), -1)

        # TensorFlow's resize method mappings
        tf_resize_methods = {
            'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            'linear': tf.image.ResizeMethod.BILINEAR,
            'bilinear': tf.image.ResizeMethod.BILINEAR,
            'bicubic': tf.image.ResizeMethod.BICUBIC,
            'area': tf.image.ResizeMethod.AREA
            # 'trilinear' and 'nearest-exact' are not directly supported in tf.image.resize
        }

        if mode in tf_resize_methods:
            output_tensor = tf.image.resize(
                images=input_tensor, size=size, method=tf_resize_methods[mode], 
                antialias=antialias
            )
        else:
            raise ValueError(f"Unsupported interpolation mode: {mode}")

        return {"interpolated_tensor": tf.squeeze(output_tensor, [0, -1]).numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.rand(3, 4).astype(np.float32),  # 2-D Tensor (H, W)
        "size": (6, 8),
        "scale_factor": None,
        "mode": 'nearest',  # You can change this to any supported mode
        "align_corners": None,
        "recompute_scale_factor": None,
        "antialias": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    if np.allclose(torch_result["interpolated_tensor"], tf_result["interpolated_tensor"], atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()