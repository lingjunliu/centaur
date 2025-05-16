import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    grid = torch.tensor(input_dict["grid"])
    mode = input_dict.get("mode", "bilinear")
    padding_mode = input_dict.get("padding_mode", "zeros")
    align_corners = input_dict.get("align_corners", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        grid = grid.cuda()
        
    result = torch.nn.functional.grid_sample(input_tensor, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    
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
        grid = tf.constant(input_dict["grid"])
        mode = input_dict.get("mode", "bilinear")
        padding_mode = input_dict.get("padding_mode", "zeros")
        align_corners = input_dict.get("align_corners", False)
        
        def tf_grid_sample(input_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
            input_shape = tf.shape(input_tensor)
            batch_size = input_shape[0]
            height = input_shape[1]
            width = input_shape[2]
            channels = input_shape[3]

            grid_shape = tf.shape(grid)
            output_height = grid_shape[1]
            output_width = grid_shape[2]

            x = tf.reshape(grid[:, :, :, 0], (batch_size, output_height, output_width))
            y = tf.reshape(grid[:, :, :, 1], (batch_size, output_height, output_width))
            
            if align_corners:
                x = (x + 1) * tf.cast((width - 1), tf.float32) / 2.0
                y = (y + 1) * tf.cast((height - 1), tf.float32) / 2.0
            else:
                x = ((x + 1) / 2.0) * tf.cast((width - 1), tf.float32)
                y = ((y + 1) / 2.0) * tf.cast((height - 1), tf.float32)

            x0 = tf.cast(tf.floor(x), tf.int32)
            y0 = tf.cast(tf.floor(y), tf.int32)
            x1 = x0 + 1
            y1 = y0 + 1

            if padding_mode == 'zeros':
                x0 = tf.clip_by_value(x0, 0, width - 1)
                x1 = tf.clip_by_value(x1, 0, width - 1)
                y0 = tf.clip_by_value(y0, 0, height - 1)
                y1 = tf.clip_by_value(y1, 0, height - 1)
            elif padding_mode == 'border':
                x0 = tf.maximum(0, tf.minimum(width - 1, x0))
                x1 = tf.maximum(0, tf.minimum(width - 1, x1))
                y0 = tf.maximum(0, tf.minimum(height - 1, y0))
                y1 = tf.maximum(0, tf.minimum(height - 1, y1))
            elif padding_mode == 'reflection':
                x0 = tf.where(x0 < 0, -x0, x0)
                x0 = tf.where(x0 > width - 1, 2 * (width - 1) - x0, x0)
                x1 = tf.where(x1 < 0, -x1, x1)
                x1 = tf.where(x1 > width - 1, 2 * (width - 1) - x1, x1)
                y0 = tf.where(y0 < 0, -y0, y0)
                y0 = tf.where(y0 > height - 1, 2 * (height - 1) - y0, y0)
                y1 = tf.where(y1 < 0, -y1, y1)
                y1 = tf.where(y1 > height - 1, 2 * (height - 1) - y1, y1)
            else:
                raise ValueError(f"Unsupported padding mode: {padding_mode}")

            x0_f = tf.cast(x0, tf.float32)
            y0_f = tf.cast(y0, tf.float32)
            x1_f = tf.cast(x1, tf.float32)
            y1_f = tf.cast(y1, tf.float32)

            def get_pixels(x, y):
                x = tf.clip_by_value(x, 0, width - 1)
                y = tf.clip_by_value(y, 0, height - 1)

                batch_range = tf.expand_dims(tf.range(batch_size), axis=1)
                base = batch_range * width * height
                base = tf.reshape(base, [batch_size, 1, 1])

                idx = base + y * width + x
                idx = tf.reshape(idx, [-1])
                pixels = tf.gather(tf.reshape(input_tensor, [-1, channels]), idx)
                return tf.reshape(pixels, [batch_size, output_height, output_width, channels])

            p00 = get_pixels(x0, y0)
            p01 = get_pixels(x0, y1)
            p10 = get_pixels(x1, y0)
            p11 = get_pixels(x1, y1)

            wa = tf.expand_dims((x1_f - x) * (y1_f - y), axis=3)
            wb = tf.expand_dims((x1_f - x) * (y - y0_f), axis=3)
            wc = tf.expand_dims((x - x0_f) * (y1_f - y), axis=3)
            wd = tf.expand_dims((x - x0_f) * (y - y0_f), axis=3)

            if mode == 'nearest':
                dist_x0 = tf.abs(x - x0_f)
                dist_y0 = tf.abs(y - y0_f)
                dist_x1 = tf.abs(x - x1_f)
                dist_y1 = tf.abs(y - y1_f)

                x_nearest = tf.where(dist_x0 < dist_x1, x0, x1)
                y_nearest = tf.where(dist_y0 < dist_y1, y0, y1)
                return get_pixels(x_nearest, y_nearest)

            elif mode == 'bilinear':
                return wa * p00 + wb * p01 + wc * p10 + wd * p11
            else:
                raise ValueError("Unsupported interpolation mode: {}".format(mode))
        

        result = tf_grid_sample(input_tensor, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32),
        "grid": np.array([[[[0.5, 0.5]]]], dtype=np.float32),
        "mode": "bilinear",
        "padding_mode": "zeros",
        "align_corners": False
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.random.rand(2, 3, 64, 64).astype(np.float32),
        "grid": np.random.rand(2, 32, 32, 2).astype(np.float32) * 2 - 1,
        "mode": "bilinear",
        "padding_mode": "zeros",
        "align_corners": False
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()