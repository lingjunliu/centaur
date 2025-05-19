import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    index_tensor = torch.tensor(input_dict["index"])
    src_tensor = torch.tensor(input_dict["src"])
    dim = input_dict.get("dim", 0)
    reduce = input_dict.get("reduce", "sum")
    include_self = input_dict.get("include_self", True)

    if not cpu:
        input_tensor = input_tensor.cuda()
        index_tensor = index_tensor.cuda()
        src_tensor = src_tensor.cuda()

    result = torch.scatter_reduce(input_tensor, dim, index_tensor, src_tensor, reduce=reduce, include_self=include_self)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    index_tensor = tf.constant(input_dict["index"])
    src_tensor = tf.constant(input_dict["src"])
    dim = input_dict.get("dim", 0)
    reduce_op = input_dict.get("reduce", "sum")
    include_self = input_dict.get("include_self", True)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        index_tensor = tf.cast(index_tensor, tf.int32)
        input_shape = tf.shape(input_tensor)
        output_shape = input_shape
        
        if reduce_op == "sum":
            result = tf.tensor_scatter_nd_add(input_tensor, tf.expand_dims(index_tensor, axis=1), src_tensor)
        elif reduce_op == "prod":
            result = tf.tensor_scatter_nd_update(tf.ones_like(input_tensor), tf.expand_dims(index_tensor, axis=1), src_tensor)
            result = input_tensor * result
        elif reduce_op == "mean":
            sums = tf.tensor_scatter_nd_add(input_tensor, tf.expand_dims(index_tensor, axis=1), src_tensor)
            counts = tf.tensor_scatter_nd_add(tf.zeros_like(input_tensor, dtype=tf.float32), tf.expand_dims(index_tensor, axis=1), tf.ones_like(src_tensor, dtype=tf.float32))
            
            if include_self:
                counts = counts + tf.ones_like(input_tensor, dtype=tf.float32)
            result = sums / counts
        elif reduce_op == "amax":
            
            scattered = tf.tensor_scatter_nd_update(tf.zeros_like(input_tensor), tf.expand_dims(index_tensor, axis=1), src_tensor)
            if include_self:
                result = tf.maximum(input_tensor, scattered)
            else:
                result = scattered
        elif reduce_op == "amin":
            scattered = tf.tensor_scatter_nd_update(tf.zeros_like(input_tensor), tf.expand_dims(index_tensor, axis=1), src_tensor)
            if include_self:
                result = tf.minimum(input_tensor, scattered)
            else:
                result = scattered
        else:
            raise ValueError(f"Unsupported reduce operation: {reduce_op}")

    return {"result": result.numpy()}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "index": np.array([0, 1, 0, 2, 1], dtype=np.int64),
        "src": np.array([6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32),
        "dim": 0,
        "reduce": "sum",
        "include_self": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()