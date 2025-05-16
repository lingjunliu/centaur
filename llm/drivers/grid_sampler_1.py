import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    grid = torch.tensor(input_dict["grid"])
    interpolation_mode = input_dict.get("interpolation_mode", 'bilinear')
    padding_mode = input_dict.get("padding_mode", 'zeros')
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
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        grid = tf.constant(input_dict["grid"])
        interpolation_mode = input_dict.get("interpolation_mode", 'bilinear')
        padding_mode = input_dict.get("padding_mode", 'zeros')
        align_corners = input_dict.get("align_corners", False)
        
        input_tensor_shape = tf.shape(input_tensor)
        grid_shape = tf.shape(grid)

        batch_size = input_tensor_shape[0]
        channels = input_tensor_shape[1]
        height = tf.cast(input_tensor_shape[2], dtype=tf.float32)
        width = tf.cast(input_tensor_shape[3], dtype=tf.float32)

        output_height = grid_shape[1]
        output_width = grid_shape[2]

        grid = tf.cast(grid, dtype=tf.float32)

        x = grid[..., 0]
        y = grid[..., 1]

        x = (x + 1) * (width - 1) / 2
        y = (y + 1) * (height - 1) / 2

        x0 = tf.floor(x)
        x1 = x0 + 1
        y0 = tf.floor(y)
        y1 = y0 + 1

        if padding_mode == 'border':
            x0 = tf.clip_by_value(x0, 0, width - 1)
            x1 = tf.clip_by_value(x1, 0, width - 1)
            y0 = tf.clip_by_value(y0, 0, height - 1)
            y1 = tf.clip_by_value(y1, 0, height - 1)
        elif padding_mode == 'zeros':
            pass
        else:
            raise ValueError(f"Padding mode {padding_mode} not implemented in tensorflow version")

        x0_safe = tf.clip_by_value(x0, 0, width - 1)
        x1_safe = tf.clip_by_value(x1, 0, width - 1)
        y0_safe = tf.clip_by_value(y0, 0, height - 1)
        y1_safe = tf.clip_by_value(y1, 0, height - 1)

        x0_valid = tf.logical_and(x0 >= 0, x0 <= width - 1)
        x1_valid = tf.logical_and(x1 >= 0, x1 <= width - 1)
        y0_valid = tf.logical_and(y0 >= 0, y0 <= height - 1)
        y1_valid = tf.logical_and(y1 >= 0, y1 <= height - 1)

        valid_mask = tf.logical_and(tf.logical_and(x0_valid, x1_valid), tf.logical_and(y0_valid, y1_valid))
        valid_mask = tf.cast(valid_mask, dtype=tf.float32)

        x0 = tf.cast(x0_safe, dtype=tf.float32)
        x1 = tf.cast(x1_safe, dtype=tf.float32)
        y0 = tf.cast(y0_safe, dtype=tf.float32)
        y1 = tf.cast(y1_safe, dtype=tf.float32)

        b = tf.range(batch_size)
        b = tf.expand_dims(b, axis=1)
        b = tf.expand_dims(b, axis=2)
        b = tf.tile(b, [1, output_height, output_width])

        b = tf.reshape(b, [-1])

        y0_flat = tf.reshape(y0, [-1])
        x0_flat = tf.reshape(x0, [-1])
        y1_flat = tf.reshape(y1, [-1])
        x1_flat = tf.reshape(x1, [-1])

        indices_a = tf.stack([b, tf.zeros_like(b, dtype=tf.int32), tf.cast(y0_flat, dtype=tf.int32), tf.cast(x0_flat, dtype=tf.int32)], axis=1)
        indices_b = tf.stack([b, tf.zeros_like(b, dtype=tf.int32), tf.cast(y1_flat, dtype=tf.int32), tf.cast(x0_flat, dtype=tf.int32)], axis=1)
        indices_c = tf.stack([b, tf.zeros_like(b, dtype=tf.int32), tf.cast(y0_flat, dtype=tf.int32), tf.cast(x1_flat, dtype=tf.int32)], axis=1)
        indices_d = tf.stack([b, tf.zeros_like(b, dtype=tf.int32), tf.cast(y1_flat, dtype=tf.int32), tf.cast(x1_flat, dtype=tf.int32)], axis=1)

        Ia = tf.gather_nd(input_tensor, tf.cast(indices_a, dtype=tf.int32))
        Ib = tf.gather_nd(input_tensor, tf.cast(indices_b, dtype=tf.int32))
        Ic = tf.gather_nd(input_tensor, tf.cast(indices_c, dtype=tf.int32))
        Id = tf.gather_nd(input_tensor, tf.cast(indices_d, dtype=tf.int32))

        wa = tf.reshape((x1 - x) * (y1 - y), [-1])
        wb = tf.reshape((x1 - x) * (y - y0), [-1])
        wc = tf.reshape((x - x0) * (y1 - y), [-1])
        wd = tf.reshape((x - x0) * (y - y0), [-1])
    
        output_flat = wa * Ia + wb * Ib + wc * Ic + wd * Id
        
        if padding_mode == 'zeros':
            valid_mask_flat = tf.reshape(valid_mask, [-1])
            output_flat = output_flat * valid_mask_flat

        output = tf.reshape(output_flat, [batch_size, output_height, output_width, channels])
        output = tf.transpose(output, perm=[0, 3, 1, 2])

        result = output.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).astype(np.float32),
        "grid": np.array([[[[-0.5, -0.5], [0.5, -0.5]], [[-0.5, 0.5], [0.5, 0.5]]]]).astype(np.float32),
        "interpolation_mode": 'bilinear',
        "padding_mode": 'zeros',
        "align_corners": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()