import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    index = torch.tensor(input_dict["index"])
    source = torch.tensor(input_dict["source"])
    dim = input_dict.get("dim", 0)
    reduce_str = input_dict["reduce"]

    if not cpu:
        input_tensor = input_tensor.cuda()
        index = index.cuda()
        source = source.cuda()

    if reduce_str == "add" or reduce_str == "sum":
        reduce = "sum"
    elif reduce_str == "multiply" or reduce_str == "prod":
        reduce = "prod"
    elif reduce_str == "minimum" or reduce_str == "amin":
        reduce = "amin"
    elif reduce_str == "maximum" or reduce_str == "amax":
        reduce = "amax"
    elif reduce_str == "mean":
        reduce = "mean"
    else:
        raise ValueError(f"Unsupported reduce operation: {reduce_str}")

    result = torch.index_reduce(input_tensor, dim, index, source, reduce=reduce)

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
        index = tf.cast(input_dict["index"], dtype=tf.int32)
        source = tf.constant(input_dict["source"])
        dim = input_dict.get("dim", 0)
        reduce = input_dict["reduce"]

        rank = len(input_tensor.shape)
        
        if dim < 0:
          dim = dim + rank

        target = tf.identity(input_tensor)

        indices = tf.unstack(index, axis=0)
        updates = tf.unstack(source, axis=0)

        if reduce == "sum" or reduce == "add":
            result = tf.tensor_scatter_nd_add(target, tf.expand_dims(indices, axis=1), updates)
        elif reduce == "mean":
            ones = tf.ones_like(index, dtype=tf.float32)
            counts = tf.tensor_scatter_nd_add(tf.zeros(tf.shape(target), dtype=tf.float32), tf.expand_dims(indices, axis=1), ones)
            result = tf.tensor_scatter_nd_update(target, tf.expand_dims(indices, axis=1), updates / counts[tf.expand_dims(indices, axis=1)])
        elif reduce == "prod" or reduce == "multiply":
            result = tf.tensor_scatter_nd_update(target, tf.expand_dims(indices, axis=1), updates)
            result = tf.math.cumprod(result, axis=dim)
        elif reduce == "amin" or reduce == "minimum":
            result = tf.tensor_scatter_nd_min(target, tf.expand_dims(indices, axis=1), updates)
        elif reduce == "amax" or reduce == "maximum":
            result = tf.tensor_scatter_nd_max(target, tf.expand_dims(indices, axis=1), updates)
        else:
            raise ValueError(f"Unsupported reduce operation: {reduce}")

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "index": np.array([0, 2, 1], dtype=np.int64),
        "source": np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=np.float32),
        "reduce": "add"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "index": np.array([0, 2], dtype=np.int64),
        "source": np.array([[10, 11, 12], [13, 14, 15]], dtype=np.float32),
        "reduce": "mean"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "index": np.array([0, 1, 2], dtype=np.int64),
        "source": np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=np.float32),
        "reduce": "minimum"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "index": np.array([0, 1, 2], dtype=np.int64),
        "source": np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=np.float32),
        "reduce": "maximum"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "index": np.array([0, 1, 2], dtype=np.int64),
        "source": np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=np.float32),
        "reduce": "prod"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()