import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    size = input_dict.get("size", None)
    scale_factor = input_dict.get("scale_factor", None)
    mode = input_dict.get("mode", 'nearest')
    align_corners = input_dict.get("align_corners", None)
    recompute_scale_factor = input_dict.get("recompute_scale_factor", None)

    if not cpu:
        input_tensor = input_tensor.cuda()

    kwargs = {}
    if size is not None:
        kwargs['size'] = size
    if scale_factor is not None:
        kwargs['scale_factor'] = scale_factor
    kwargs['mode'] = mode
    if mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] and align_corners is not None:
        kwargs['align_corners'] = align_corners
    if recompute_scale_factor is not None:
        kwargs['recompute_scale_factor'] = recompute_scale_factor

    m = torch.nn.Upsample(**kwargs)
    result = m(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_np = input_dict["input"]
    size = input_dict.get("size", None)
    scale_factor = input_dict.get("scale_factor", None)
    mode = input_dict.get("mode", 'nearest')
    align_corners = input_dict.get("align_corners", False)
    recompute_scale_factor = input_dict.get("recompute_scale_factor", None)

    input_tensor = tf.constant(input_np)
    input_shape = input_tensor.shape

    if len(input_shape) == 3:
        input_tensor = tf.expand_dims(input_tensor, axis=0)
    elif len(input_shape) == 2:
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        input_tensor = tf.expand_dims(input_tensor, axis=0)

    if scale_factor is not None:
        if isinstance(scale_factor, float) or isinstance(scale_factor, int):
            scale_factor = [scale_factor] * (len(input_shape) - 2)

        if isinstance(scale_factor, list) or isinstance(scale_factor, tuple):
            scale_factor = list(scale_factor)

        if len(scale_factor) != len(input_shape) - 2:
            raise ValueError("scale_factor must match input size")

        new_size = []
        for i in range(len(input_shape) - 2):
            new_size.append(int(input_shape[i + 2] * scale_factor[i]))
        size = new_size

    if size is not None:
        size = list(size)
        if len(size) != len(input_shape) - 2:
            raise ValueError("size must match input size")

        if len(input_shape) == 4:
            if mode == 'nearest':
                method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
            elif mode == 'bilinear':
                method = tf.image.ResizeMethod.BILINEAR
            else:
                method = tf.image.ResizeMethod.NEAREST_NEIGHBOR

            result = tf.image.resize(input_tensor, size, method=method, antialias=False)

        elif len(input_shape) == 5:
            raise NotImplementedError("3D Upsample not implemented")
    else:
        result = input_tensor

    if len(result.shape) == 4 and len(input_np.shape) == 4:
        if result.shape[1] != input_np.shape[1]:
            result = tf.transpose(result, perm=[0, 2, 3, 1])
            result = tf.expand_dims(result, axis=1)

    elif len(result.shape) != len(input_np.shape):
        result = tf.squeeze(result, axis=0)

    result = result.numpy()
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2),
        "scale_factor": 2,
        "mode": 'nearest'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2),
        "scale_factor": 2,
        "mode": 'bilinear',
        "align_corners": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2),
        "scale_factor": 2,
        "mode": 'bilinear',
        "align_corners": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.zeros((3, 3), dtype=np.float32).reshape(1, 1, 3, 3),
        "scale_factor": 2,
        "mode": 'bilinear',
        "align_corners": False
    }
    input_data["input"][:, :, :2, :2] = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2)

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.zeros((3, 3), dtype=np.float32).reshape(1, 1, 3, 3),
        "scale_factor": 2,
        "mode": 'bilinear',
        "align_corners": True
    }
    input_data["input"][:, :, :2, :2] = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2)

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()