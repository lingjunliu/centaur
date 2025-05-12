import numpy as np
import torch

def torch_version(input_dict, cpu=True):
    input_tensor = torch.tensor(input_dict['input'])
    grid_tensor = torch.tensor(input_dict['grid'])
    interpolation_mode = input_dict.get("interpolation_mode", "bilinear")
    padding_mode = input_dict.get("padding_mode", "zeros")
    align_corners = input_dict.get("align_corners", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        grid_tensor = grid_tensor.cuda()
    
    result = torch.nn.functional.grid_sample(input_tensor, grid_tensor, mode=interpolation_mode, padding_mode=padding_mode, align_corners=align_corners)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.convert_to_tensor(input_dict['input'], dtype=tf.float32)
        grid_tensor = tf.convert_to_tensor(input_dict['grid'], dtype=tf.float32)

        input_shape = input_tensor.shape
        grid_shape = grid_tensor.shape
        
        batch_size = tf.compat.v1.dimension_value(input_shape[0]) or tf.shape(input_tensor)[0]
        in_height = tf.compat.v1.dimension_value(input_shape[2]) or tf.shape(input_tensor)[2]
        in_width = tf.compat.v1.dimension_value(input_shape[3]) or tf.shape(input_tensor)[3]
        out_height = tf.compat.v1.dimension_value(grid_shape[1]) or tf.shape(grid_tensor)[1]
        out_width = tf.compat.v1.dimension_value(grid_shape[2]) or tf.shape(grid_tensor)[2]
        num_channels = tf.compat.v1.dimension_value(input_shape[1]) or tf.shape(input_tensor)[1]
        
        x = grid_tensor[..., 0]
        y = grid_tensor[..., 1]

        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)

        x = tf.clip_by_value(x, -1.0, 1.0)
        y = tf.clip_by_value(y, -1.0, 1.0)
        
        x = (x + 1.0) / 2.0 * (in_width - 1)
        y = (y + 1.0) / 2.0 * (in_height - 1)

        x0 = tf.floor(x)
        x1 = tf.clip_by_value(x0 + 1, 0, in_width - 1)
        y0 = tf.floor(y)
        y1 = tf.clip_by_value(y0 + 1, 0, in_height - 1)

        x0 = tf.cast(x0, tf.int32)
        x1 = tf.cast(x1, tf.int32)
        y0 = tf.cast(y0, tf.int32)
        y1 = tf.cast(y1, tf.int32)

        def _get_pixels_value(input_tensor, batch_idx, y_idx, x_idx):
            indices = tf.stack([batch_idx, y_idx, x_idx], axis=-1)
            return tf.gather_nd(input_tensor[0], indices)

        def _bilinear_interpolation(input_tensor, x, y):
            x0_f = tf.floor(x)
            y0_f = tf.floor(y)
            x1_f = x0_f + 1.0
            y1_f = y0_f + 1.0

            x0 = tf.cast(x0_f, tf.int32)
            x1 = tf.cast(tf.minimum(x1_f, float(in_width - 1)), tf.int32)
            y0 = tf.cast(y0_f, tf.int32)
            y1 = tf.cast(tf.minimum(y1_f, float(in_height - 1)), tf.int32)

            Ia = _get_pixels_value(input_tensor, 0, y0, x0)
            Ib = _get_pixels_value(input_tensor, 0, y1, x0)
            Ic = _get_pixels_value(input_tensor, 0, y0, x1)
            Id = _get_pixels_value(input_tensor, 0, y1, x1)

            wa = (x1_f - x) * (y1_f - y)
            wb = (x1_f - x) * (y - y0_f)
            wc = (x - x0_f) * (y1_f - y)
            wd = (x - x0_f) * (y - y0_f)

            return wa * tf.cast(Ia, tf.float32) + wb * tf.cast(Ib, tf.float32) + wc * tf.cast(Ic, tf.float32) + wd * tf.cast(Id, tf.float32)

        sampled_pixels = _bilinear_interpolation(input_tensor, x, y)
        result = tf.expand_dims(sampled_pixels, axis=0)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[[1.0, 2.0], [3.0, 4.0]]]]).astype(np.float32),
        "grid": np.array([[[[0.5, 0.5]]]]).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[[[1.0, 2.0], [3.0, 4.0]]]]).astype(np.float32),
        "grid": np.array([[[[0.75, 0.75]]]]).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[[[1.0, 2.0], [3.0, 4.0]]]]).astype(np.float32),
        "grid": np.array([[[[-0.5, -0.5]]]]).astype(np.float32)
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()