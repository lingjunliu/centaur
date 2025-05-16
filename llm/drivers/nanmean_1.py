import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

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
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        dim = input_dict.get("dim", None)
        keepdim = input_dict.get("keepdim", False)
        dtype = input_dict.get("dtype", None)

        if dtype is not None:
            input_tensor = tf.cast(input_tensor, dtype)

        mask = tf.math.is_finite(input_tensor)
        masked_tensor = tf.where(mask, input_tensor, tf.zeros_like(input_tensor))
        
        if dim is None:
            num_valid = tf.cast(tf.reduce_sum(tf.cast(mask, tf.float32)), tf.float32)
            result = tf.reduce_sum(masked_tensor) / num_valid
        else:
            num_valid = tf.cast(tf.reduce_sum(tf.cast(mask, tf.float32), axis=dim, keepdims=True), tf.float32)
            sum_masked = tf.reduce_sum(masked_tensor, axis=dim, keepdims=True)
            result = sum_masked / num_valid
            
            if not keepdim:
                if isinstance(dim, int):
                    result = tf.squeeze(result, axis=dim)
                else:
                    for i, d in enumerate(sorted(dim, reverse=True)):
                        result = tf.squeeze(result, axis=d)

        if tf.reduce_sum(tf.cast(tf.math.is_nan(result), tf.int32)).numpy() > 0:
            result = tf.constant(np.nan, dtype=result.dtype)
    
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

    print("Success")

if __name__ == "__main__":
    main()