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
    recompute_scale_factor = input_dict.get("recompute_scale_factor", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.functional.interpolate(input_tensor, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        size = input_dict.get("size", None)
        scale_factor = input_dict.get("scale_factor", None)
        mode = input_dict.get("mode", 'nearest')
        align_corners = input_dict.get("align_corners", None)
        
        input_shape = input_tensor.shape
        
        if scale_factor is not None:
          if isinstance(scale_factor, float):
            scale_factor = [scale_factor] * (len(input_shape) - 2)
          scale_factor = tf.constant(scale_factor, dtype=tf.float32)
          
          new_shape = tf.cast(tf.round(tf.cast(input_shape[1:-1], tf.float32) * scale_factor), tf.int32)
          size = new_shape
        
        if size is not None:
          size = tf.cast(size, tf.int32)
          if len(input_shape) == 3:
            size = [size[0]]
          
          if mode == 'nearest':
            method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
          elif mode == 'linear' or mode == 'bilinear':
            method = tf.image.ResizeMethod.BILINEAR
          elif mode == 'bicubic':
            method = tf.image.ResizeMethod.BICUBIC
          else:
              raise ValueError(f"Unsupported mode: {mode}")

          if len(input_shape) == 4:
              result = tf.image.resize(input_tensor, size, method=method, preserve_aspect_ratio=False)
          elif len(input_shape) == 3:
              result = tf.image.resize(tf.expand_dims(input_tensor, axis=0), size, method=method, preserve_aspect_ratio=False)
              result = tf.squeeze(result, axis=0)
          elif len(input_shape) == 2:
              result = tf.image.resize(tf.expand_dims(tf.expand_dims(input_tensor, axis=0), axis=-1), [size[0], 1], method=method, preserve_aspect_ratio=False)
              result = tf.squeeze(result, axis=0)
              result = tf.squeeze(result, axis=-1)
          else:
              raise ValueError(f"Unsupported input shape: {input_shape}")

        result = result.numpy()
        
        if len(result.shape) < len(input_tensor.shape):
            result = np.expand_dims(result, axis=0)
        
        if result.shape != torch_result["result"].shape:
            if len(input_shape) == 4:
                expected_shape = torch_result["result"].shape
                result = tf.image.resize(input_tensor, expected_shape[2:4], method=method, preserve_aspect_ratio=False).numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 1, 10, 10).astype(np.float32),
        "scale_factor": [2.0, 2.0],
        "mode": 'nearest',
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(1, 1, 10, 10).astype(np.float32),
        "size": [20, 20],
        "mode": 'nearest',
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.random.rand(1, 10, 10).astype(np.float32),
        "size": [20, 20],
        "mode": 'nearest',
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(1, 1, 10).astype(np.float32),
        "size": [20],
        "mode": 'nearest',
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(10, 10).astype(np.float32),
        "size": [20, 20],
        "mode": 'nearest',
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(1, 1, 5, 5).astype(np.float32),
        "size": [10, 10],
        "mode": 'nearest',
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()