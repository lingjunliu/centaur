import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    size = input_dict.get("size", None)
    scale_factor = input_dict.get("scale_factor", None)
    mode = input_dict.get("mode", 'nearest')
    align_corners = input_dict.get("align_corners", None)
    recompute_scale_factor = input_dict.get("recompute_scale_factor", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    upsample = torch.nn.Upsample(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
    result = upsample(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    size = input_dict.get("size", None)
    scale_factor = input_dict.get("scale_factor", None)
    mode = input_dict.get("mode", 'nearest')
    align_corners = input_dict.get("align_corners", None)
    recompute_scale_factor = input_dict.get("recompute_scale_factor", False)

    input_shape = input_tensor.shape
    if len(input_shape) == 3:
        input_tensor = tf.expand_dims(input_tensor, axis=0)
    elif len(input_shape) == 2:
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        input_tensor = tf.expand_dims(input_tensor, axis=0)

    if scale_factor is not None:
        if isinstance(scale_factor, (int, float)):
            scale_factor = [float(scale_factor)] * (len(input_shape) - 2)
        scale_factor = list(scale_factor)
    elif size is not None:
        scale_factor = [s / float(i) for i, s in zip(input_tensor.shape[2:], size)]
    else:
        raise ValueError("Either size or scale_factor must be specified")

    if mode == 'nearest':
        method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    elif mode == 'linear' or mode == 'bilinear':
        method = tf.image.ResizeMethod.BILINEAR
    elif mode == 'bicubic':
        method = tf.image.ResizeMethod.BICUBIC
    elif mode == 'trilinear':
        method = tf.image.ResizeMethod.BILINEAR
    else:
        raise ValueError("Unsupported mode: {}".format(mode))

    if size is None:
        new_size = [int(input_tensor.shape[i] * scale_factor[i-2]) for i in range(2, len(input_tensor.shape))]
    else:
        new_size = size

    result = tf.image.resize(input_tensor, new_size, method=method)

    if align_corners is not None:
        if mode == 'bilinear' and align_corners:
            scale_height = (new_size[0] - 1) / (float(input_tensor.shape[2] - 1))
            scale_width = (new_size[1] - 1) / (float(input_tensor.shape[3] - 1))

            height_indices = tf.clip_by_value(tf.round(tf.range(new_size[0]) / scale_height), 0, input_tensor.shape[2]-1)
            width_indices = tf.clip_by_value(tf.round(tf.range(new_size[1]) / scale_width), 0, input_tensor.shape[3]-1)

            height_indices = tf.cast(height_indices, dtype=tf.int32)
            width_indices = tf.cast(width_indices, dtype=tf.int32)
            
            result_list = []
            for h in range(new_size[0]):
                row_list = []
                for w in range(new_size[1]):
                    row_list.append(result[0, :, height_indices[h], width_indices[w]])
                result_list.append(tf.stack(row_list, axis=1))
            result = tf.stack(result_list, axis=1)
            result = tf.expand_dims(result, axis=0)

    if len(input_shape) == 3:
        result = tf.squeeze(result, axis=0)
    if len(input_shape) == 2:
        result = tf.squeeze(result, axis=0)
        result = tf.squeeze(result, axis=0)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 1, 10, 10).astype(np.float32),
        "scale_factor": 2.0,
        "mode": 'nearest'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(1, 1, 10, 10).astype(np.float32),
        "size": [20, 20],
        "mode": 'nearest'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(1, 1, 10, 10).astype(np.float32),
        "scale_factor": 2.0,
        "mode": 'bilinear',
        "align_corners": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(1, 1, 10, 10).astype(np.float32),
        "size": [20, 20],
        "mode": 'bilinear',
        "align_corners": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(1, 1, 10, 10).astype(np.float32),
        "scale_factor": 2.0,
        "mode": 'bilinear',
        "align_corners": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(1, 1, 10, 10).astype(np.float32),
        "size": [20, 20],
        "mode": 'bilinear',
        "align_corners": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()