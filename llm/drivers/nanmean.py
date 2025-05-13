import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    dim = input_dict.get("dim", None)
    keepdim = input_dict.get("keepdim", False)
    dtype = input_dict.get("dtype", None)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nanmean(input_tensor, dim=dim, keepdim=keepdim, dtype=dtype)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    dim = input_dict.get("dim", None)
    keepdim = input_dict.get("keepdim", False)
    dtype = input_dict.get("dtype", None)

    if dtype is not None:
        if dtype == 'torch.float64':
            input_tensor = tf.cast(input_tensor, tf.float64)

    if not cpu:
        with tf.device('/GPU:0'):
            if dim is None:
                mask = tf.math.is_finite(input_tensor)
                masked_tensor = tf.boolean_mask(input_tensor, mask)
                if tf.size(masked_tensor) == 0:
                    result = tf.constant(np.nan, dtype=input_tensor.dtype)
                else:
                    result = tf.reduce_mean(masked_tensor)
            elif isinstance(dim, int):
                result_list = []
                for i in range(tf.shape(input_tensor)[dim]):
                    
                    slice_tensor = tf.gather(input_tensor, i, axis=dim)

                    mask = tf.math.is_finite(slice_tensor)
                    masked_tensor = tf.boolean_mask(slice_tensor, mask)

                    if tf.size(masked_tensor) == 0:
                        result_list.append(tf.constant(np.nan, dtype=input_tensor.dtype))
                    else:
                        result_list.append(tf.reduce_mean(masked_tensor))
                
                result = tf.stack(result_list)
                if keepdim:
                    shape = list(input_tensor.shape)
                    shape[dim] = 1
                    result = tf.reshape(result, shape)
            else:
                input_tensor_np = input_tensor.numpy()
                dim = tuple(dim)
                masked_array = np.ma.masked_array(input_tensor_np, mask=np.isnan(input_tensor_np))
                result_np = np.ma.mean(masked_array, axis=dim, keepdims=keepdim).filled(np.nan)
                result = tf.constant(result_np, dtype=input_tensor.dtype)

    else:
        if dim is None:
            mask = tf.math.is_finite(input_tensor)
            masked_tensor = tf.boolean_mask(input_tensor, mask)
            if tf.size(masked_tensor) == 0:
                result = tf.constant(np.nan, dtype=input_tensor.dtype)
            else:
                result = tf.reduce_mean(masked_tensor)
        elif isinstance(dim, int):
            result_list = []
            for i in range(tf.shape(input_tensor)[dim]):
                slice_tensor = tf.gather(input_tensor, i, axis=dim)

                mask = tf.math.is_finite(slice_tensor)
                masked_tensor = tf.boolean_mask(slice_tensor, mask)

                if tf.size(masked_tensor) == 0:
                    result_list.append(tf.constant(np.nan, dtype=input_tensor.dtype))
                else:
                    result_list.append(tf.reduce_mean(masked_tensor))

            result = tf.stack(result_list)

            if keepdim:
                shape = list(input_tensor.shape)
                shape[dim] = 1
                result = tf.reshape(result, shape)
        else:
            input_tensor_np = input_tensor.numpy()
            dim = tuple(dim)
            masked_array = np.ma.masked_array(input_tensor_np, mask=np.isnan(input_tensor_np))
            result_np = np.ma.mean(masked_array, axis=dim, keepdims=keepdim).filled(np.nan)
            result = tf.constant(result_np, dtype=input_tensor.dtype)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[np.nan, 1, 2], [1, 2, 3]], dtype=np.float32),
        "dim": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL, equal_nan=True), "Results do not match"

    input_data = {
        "input": np.array([np.nan], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL, equal_nan=True), "Results do not match"
    
    input_data = {
        "input": np.array([[np.nan, 1, 2], [1, 2, 3]], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL, equal_nan=True), "Results do not match"
    
    input_data = {
        "input": np.array([[np.nan, 1, 2], [1, 2, 3]], dtype=np.float32),
        "dim": 0,
        "keepdim": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL, equal_nan=True), "Results do not match"
    
    input_data = {
        "input": np.array([[[np.nan, 1, 2], [1, 2, 3]],[[np.nan, 1, 2], [1, 2, 3]]], dtype=np.float32),
        "dim": (0,1),
        "keepdim": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL, equal_nan=True), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()