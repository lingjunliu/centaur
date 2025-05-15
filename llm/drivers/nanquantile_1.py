import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    q = torch.tensor(input_dict["q"]) if isinstance(input_dict["q"], np.ndarray) else input_dict["q"]
    dim = input_dict.get("dim", None)
    keepdim = input_dict.get("keepdim", False)
    interpolation = input_dict.get("interpolation", 'linear')

    if not cpu:
        input_tensor = input_tensor.cuda()
        if isinstance(q, torch.Tensor):
            q = q.cuda()

    result = torch.nanquantile(input_tensor, q, dim=dim, keepdim=keepdim, interpolation=interpolation)

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
        q = tf.constant(input_dict["q"]) if isinstance(input_dict["q"], np.ndarray) else input_dict["q"]
        dim = input_dict.get("dim", None)
        keepdim = input_dict.get("keepdim", False)
        interpolation = input_dict.get("interpolation", 'linear')

        input_tensor_shape = tf.shape(input_tensor)

        if dim is None:
            input_tensor_flat = tf.reshape(input_tensor, [-1])
            valid_values = tf.boolean_mask(input_tensor_flat, tf.math.logical_not(tf.math.is_nan(input_tensor_flat)))
            if tf.size(valid_values) == 0:
                result = tf.constant(np.nan, dtype=input_tensor.dtype) if isinstance(q, float) or isinstance(q, int) else tf.fill(tf.shape(q), np.nan)

            else:
                result = quantile(valid_values, q, interpolation)
        else:
            if dim < 0:
                dim = dim + len(input_tensor.shape)
            
            result_list = []
            
            for i in range(input_tensor_shape[dim]):
                indices = [slice(None)] * len(input_tensor.shape)
                indices[dim] = i
                slice_indices = tuple(indices)

                tensor_slice = input_tensor[slice_indices]

                valid_values = tf.boolean_mask(tf.reshape(tensor_slice, [-1]), tf.math.logical_not(tf.math.is_nan(tf.reshape(tensor_slice, [-1]))))
                if tf.size(valid_values) == 0:
                     quantile_result = tf.constant(np.nan, dtype=input_tensor.dtype) if isinstance(q, float) or isinstance(q, int) else tf.fill(tf.shape(q), np.nan)
                else:
                    quantile_result = quantile(valid_values, q, interpolation)
                result_list.append(quantile_result)

            result = tf.stack(result_list, axis=dim)

            if keepdim:
                result = tf.expand_dims(result, axis=dim)

        result = result.numpy()
    return {"result": result}


def quantile(valid_values, q, interpolation):
    import tensorflow as tf

    valid_values = tf.sort(valid_values)
    n = tf.cast(tf.size(valid_values), dtype=tf.float32)

    if isinstance(q, float) or isinstance(q, int):
        rank = q * (n - 1)
        rank_0 = tf.clip_by_value(tf.floor(rank), 0, n - 1)
        rank_1 = tf.clip_by_value(tf.minimum(rank_0 + 1, n - 1), 0, n - 1)
        weight_1 = rank - rank_0
        weight_0 = 1 - weight_1

        if interpolation == 'linear':
            result = weight_0 * tf.gather(valid_values, tf.cast(rank_0, dtype=tf.int32)) + weight_1 * tf.gather(valid_values, tf.cast(rank_1, dtype=tf.int32))
        elif interpolation == 'lower':
            result = tf.gather(valid_values, tf.cast(rank_0, dtype=tf.int32))
        elif interpolation == 'higher':
            result = tf.gather(valid_values, tf.cast(rank_1, dtype=tf.int32))
        elif interpolation == 'midpoint':
            result = (tf.gather(valid_values, tf.cast(rank_0, dtype=tf.int32)) + tf.gather(valid_values, tf.cast(rank_1, dtype=tf.int32))) / 2.0
        elif interpolation == 'nearest':
            result = tf.gather(valid_values, tf.cast(tf.round(rank), dtype=tf.int32))
        else:
            raise ValueError(f"Invalid interpolation method: {interpolation}")
        return result

    else:
        q = tf.cast(q, dtype=tf.float32)
        result = []
        for quantile_value in q.numpy():
            rank = quantile_value * (n - 1)
            rank_0 = tf.clip_by_value(tf.floor(rank), 0, n - 1)
            rank_1 = tf.clip_by_value(tf.minimum(rank_0 + 1, n - 1), 0, n - 1)
            weight_1 = rank - rank_0
            weight_0 = 1 - weight_1
            
            if interpolation == 'linear':
                res = weight_0 * tf.gather(valid_values, tf.cast(rank_0, dtype=tf.int32)) + weight_1 * tf.gather(valid_values, tf.cast(rank_1, dtype=tf.int32))
            elif interpolation == 'lower':
                res = tf.gather(valid_values, tf.cast(rank_0, dtype=tf.int32))
            elif interpolation == 'higher':
                res = tf.gather(valid_values, tf.cast(rank_1, dtype=tf.int32))
            elif interpolation == 'midpoint':
                res = (tf.gather(valid_values, tf.cast(rank_0, dtype=tf.int32)) + tf.gather(valid_values, tf.cast(rank_1, dtype=tf.int32))) / 2.0
            elif interpolation == 'nearest':
                res = tf.gather(valid_values, tf.cast(tf.round(rank), dtype=tf.int32))
            else:
                raise ValueError(f"Invalid interpolation method: {interpolation}")
            result.append(res)
        return tf.stack(result)

def main():
    import torch
    A_TOL = 0.01

    input_data = {
        "input": np.array([float('nan'), 1, 2, 3, 4, 5], dtype=np.float32),
        "q": 0.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL, equal_nan=True), "Results do not match"

    input_data = {
        "input": np.array([[float('nan'), float('nan')], [1, 2]], dtype=np.float32),
        "q": 0.5,
        "dim": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL, equal_nan=True), "Results do not match"

    input_data = {
        "input": np.array([[float('nan'), float('nan')], [1, 2]], dtype=np.float32),
        "q": 0.5,
        "dim": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], equal_nan=True, atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([float('nan'), 1, 2], dtype=np.float32),
        "q": np.array([0.25, 0.5, 0.75], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL, equal_nan=True), "Results do not match"

    input_data = {
        "input": np.array([[1.0, 2.0, float('nan')], [4.0, float('nan'), 6.0]], dtype=np.float32),
        "q": 0.5,
        "dim": 1,
        "interpolation": "nearest"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], equal_nan=True, atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1.0, 2.0, float('nan')], [4.0, float('nan'), 6.0]], dtype=np.float32),
        "q": 0.5,
        "dim": 1,
        "interpolation": "midpoint"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], equal_nan=True, atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1.0, 2.0, float('nan')], [4.0, float('nan'), 6.0]], dtype=np.float32),
        "q": 0.5,
        "dim": 1,
        "interpolation": "lower"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], equal_nan=True, atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([[1.0, 2.0, float('nan')], [4.0, float('nan'), 6.0]], dtype=np.float32),
        "q": 0.5,
        "dim": 1,
        "interpolation": "higher"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], equal_nan=True, atol=A_TOL), "Results do not match"
    
    print("Success")


if __name__ == "__main__":
    main()