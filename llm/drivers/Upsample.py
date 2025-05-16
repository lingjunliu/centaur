import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    size = input_dict.get("size", None)
    scale_factor = input_dict.get("scale_factor", None)
    mode = input_dict.get("mode", 'nearest')
    align_corners = input_dict.get("align_corners", None)
    recompute_scale_factor = input_dict.get("recompute_scale_factor", None)

    if not cpu:
        input_tensor = input_tensor.cuda()

    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    
    upsample = torch.nn.Upsample(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
    result = upsample(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.squeeze(0).squeeze(0).numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import tensorflow.keras.layers as layers

    input_tensor = tf.constant(input_dict["input"])
    size = input_dict.get("size", None)
    scale_factor = input_dict.get("scale_factor", None)
    mode = input_dict.get("mode", 'nearest')
    align_corners = input_dict.get("align_corners", None)

    if mode not in ['nearest', 'bilinear', 'bicubic']:
        raise ValueError(f"Unsupported mode: {mode}")
    
    if not cpu:
        device_string = "/GPU:0"
    else:
        device_string = "/CPU:0"
    
    with tf.device(device_string):
        input_shape = input_tensor.shape
        input_tensor = tf.reshape(input_tensor, (1, input_shape[0], input_shape[1], 1))
        
        if size is not None:
            new_height = size[0]
            new_width = size[1]
            
        elif scale_factor is not None:
            if isinstance(scale_factor, float) or isinstance(scale_factor, int):
                new_height = int(input_shape[0] * scale_factor)
                new_width = int(input_shape[1] * scale_factor)
            else:
                new_height = int(input_shape[0] * scale_factor[0])
                new_width = int(input_shape[1] * scale_factor[1])
        else:
             raise ValueError("Either size or scale_factor must be specified.")

        if mode == 'nearest':
            method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        elif mode == 'bilinear':
            method = tf.image.ResizeMethod.BILINEAR
        elif mode == 'bicubic':
            method = tf.image.ResizeMethod.BICUBIC
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if align_corners is None or align_corners == False:
            align_corners_tf = False
        else:
            align_corners_tf = True
        
        result = tf.image.resize(input_tensor, [new_height, new_width], method=method, antialias=False, preserve_aspect_ratio=False)

        result = tf.reshape(result, (new_height, new_width))
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "scale_factor": 2.0,
        "mode": "nearest"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "size": (4,4),
        "mode": "nearest"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "scale_factor": 2.0,
        "mode": "bilinear",
        "align_corners": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()