import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    grid = torch.tensor(input_dict["grid"])
    interpolation_mode = input_dict.get("interpolation_mode", "bilinear")
    padding_mode = input_dict.get("padding_mode", "zeros")
    align_corners = input_dict.get("align_corners", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        grid = grid.cuda()

    result = torch.nn.functional.grid_sample(input_tensor, grid, mode=interpolation_mode, padding_mode=padding_mode, align_corners=align_corners)

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
        input_tensor = tf.constant(input_dict["input"])
        grid = tf.constant(input_dict["grid"])
        interpolation_mode = input_dict.get("interpolation_mode", "bilinear")
        padding_mode = input_dict.get("padding_mode", "zeros")
        align_corners = input_dict.get("align_corners", False)

        def tf_grid_sample(input_tensor, grid, interpolation_mode='bilinear', padding_mode='zeros', align_corners=False):
            input_shape = tf.shape(input_tensor)
            batch_size = input_shape[0]
            height = tf.cast(input_shape[1], tf.float32)
            width = tf.cast(input_shape[2], tf.float32)
            channels = input_shape[3]

            grid_shape = tf.shape(grid)
            output_height = grid_shape[1]
            output_width = grid_shape[2]

            x = tf.reshape(grid[..., 0], [-1])
            y = tf.reshape(grid[..., 1], [-1])

            x = (x + 1) / 2 * (width - 1)
            y = (y + 1) / 2 * (height - 1)

            x0 = tf.floor(x)
            x1 = x0 + 1
            y0 = tf.floor(y)
            y1 = y0 + 1

            if padding_mode == 'zeros':
                x0_safe = tf.clip_by_value(x0, 0, width - 1)
                x1_safe = tf.clip_by_value(x1, 0, width - 1)
                y0_safe = tf.clip_by_value(y0, 0, height - 1)
                y1_safe = tf.clip_by_value(y1, 0, height - 1)

                x0_valid_mask = tf.logical_and(x0 >= 0, x0 <= width - 1)
                x1_valid_mask = tf.logical_and(x1 >= 0, x1 <= width - 1)
                y0_valid_mask = tf.logical_and(y0 >= 0, y0 <= height - 1)
                y1_valid_mask = tf.logical_and(y1 >= 0, y1 <= height - 1)

                valid_mask = tf.logical_and(tf.logical_and(x0_valid_mask, x1_valid_mask),
                                             tf.logical_and(y0_valid_mask, y1_valid_mask))

                x0 = x0_safe
                x1 = x1_safe
                y0 = y0_safe
                y1 = y1_safe
            elif padding_mode == 'border':
                x0 = tf.clip_by_value(x0, 0, width - 1)
                x1 = tf.clip_by_value(x1, 0, width - 1)
                y0 = tf.clip_by_value(y0, 0, height - 1)
                y1 = tf.clip_by_value(y1, 0, height - 1)
                valid_mask = tf.ones_like(x0, dtype=tf.bool)
            elif padding_mode == 'reflection':
                x0 = tf.where(x0 < 0, -x0, tf.where(x0 > width - 1, 2 * (width - 1) - x0, x0))
                x1 = tf.where(x1 < 0, -x1, tf.where(x1 > width - 1, 2 * (width - 1) - x1, x1))
                y0 = tf.where(y0 < 0, -y0, tf.where(y0 > height - 1, 2 * (height - 1) - y0, y0))
                y1 = tf.where(y1 < 0, -y1, tf.where(y1 > height - 1, 2 * (height - 1) - y1, y1))
                valid_mask = tf.ones_like(x0, dtype=tf.bool)
            else:
                raise ValueError("Invalid padding mode: {}".format(padding_mode))

            x0_int = tf.cast(x0, tf.int32)
            x1_int = tf.cast(x1, tf.int32)
            y0_int = tf.cast(y0, tf.int32)
            y1_int = tf.cast(y1, tf.int32)

            def _get_pixels(x_idx, y_idx):
                indices = tf.stack([y_idx, x_idx], axis=-1)
                gathered = tf.gather_nd(tf.reshape(input_tensor, [tf.cast(height * width, tf.int32), channels]), indices)
                return tf.cast(gathered, tf.float32)  

            pixels_00 = _get_pixels(x0_int, y0_int)
            pixels_01 = _get_pixels(x0_int, y1_int)
            pixels_10 = _get_pixels(x1_int, y0_int)
            pixels_11 = _get_pixels(x1_int, y1_int)

            if interpolation_mode == 'bilinear':
                wa = tf.expand_dims((x1 - x) * (y1 - y), axis=1)
                wb = tf.expand_dims((x1 - x) * (y - y0), axis=1)
                wc = tf.expand_dims((x - x0) * (y1 - y), axis=1)
                wd = tf.expand_dims((x - x0) * (y - y0), axis=1)

                output_pixels = tf.where(
                    tf.expand_dims(valid_mask, axis=1),
                    wa * pixels_00 + wb * pixels_01 + wc * pixels_10 + wd * pixels_11,
                    tf.zeros_like(pixels_00)
                )
            elif interpolation_mode == 'nearest':
                x_round = tf.round(x)
                y_round = tf.round(y)

                x_round_int = tf.cast(x_round, tf.int32)
                y_round_int = tf.cast(y_round, tf.int32)

                x_round_int = tf.clip_by_value(x_round_int, 0, tf.cast(width - 1, tf.int32))
                y_round_int = tf.clip_by_value(y_round_int, 0, tf.cast(height - 1, tf.int32))

                output_pixels = _get_pixels(x_round_int, y_round_int)
            else:
                raise ValueError("Invalid interpolation mode: {}".format(interpolation_mode))
            
            
            
            output_shape = tf.stack([batch_size, output_height, output_width, channels])
            output = tf.reshape(output_pixels, output_shape)
            return output

        result = tf_grid_sample(input_tensor, grid, interpolation_mode=interpolation_mode, padding_mode=padding_mode, align_corners=align_corners)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32),
        "grid": np.array([[[[0.5, 0.5]]]], dtype=np.float32),
        "interpolation_mode": "bilinear",
        "padding_mode": "zeros",
        "align_corners": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()