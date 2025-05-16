import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    index = torch.tensor(input_dict["index"])
    source = torch.tensor(input_dict["source"])
    dim = input_dict.get("dim", 0)
    alpha = input_dict.get("alpha", 1.0)
    reduce = input_dict["reduce"]

    if not cpu:
        input_tensor = input_tensor.cuda()
        index = index.cuda()
        source = source.cuda()

    if reduce == "sum":
      result = torch.index_reduce(input_tensor, dim, index, source, reduce="add")
    elif reduce == "prod":
      result = torch.index_reduce(input_tensor, dim, index, source, reduce="multiply")
    elif reduce == "mean":
      result = torch.index_reduce(input_tensor, dim, index, source, reduce="mean")
    elif reduce == "amax":
      result = torch.index_reduce(input_tensor, dim, index, source, reduce="amax")
    elif reduce == "amin":
      result = torch.index_reduce(input_tensor, dim, index, source, reduce="amin")
    else:
      result = torch.index_reduce(input_tensor, dim, index, source, reduce=reduce)

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
        index = tf.cast(tf.constant(input_dict["index"]), dtype=tf.int32)
        source = tf.constant(input_dict["source"])
        dim = input_dict.get("dim", 0)
        alpha = input_dict.get("alpha", 1.0)
        reduce_op = input_dict["reduce"]

        input_shape = tf.shape(input_tensor)
        num_elements = tf.reduce_prod(input_shape[:dim])
        outer_shape = tf.shape(input_tensor)[dim+1:]
        num_outer = tf.reduce_prod(outer_shape)

        input_reshaped = tf.reshape(input_tensor, [num_elements, tf.shape(input_tensor)[dim], num_outer])
        source_reshaped = tf.reshape(source, [num_elements, tf.shape(source)[dim], num_outer])

        output_list = []
        for i in range(num_elements):
            output_inner_list = []
            for j in range(num_outer):
                input_slice = input_reshaped[i, :, j]
                source_slice = source_reshaped[i, :, j]
                index_slice = index

                if reduce_op == "sum":
                    target = tf.tensor_scatter_nd_add(input_slice, tf.expand_dims(index_slice, axis=1), source_slice * alpha)
                    output_slice = target
                elif reduce_op == "prod":
                    updates = tf.tensor_scatter_nd_update(tf.ones_like(input_slice), tf.expand_dims(index_slice, axis=1), tf.gather_nd(source_slice * alpha, tf.expand_dims(index_slice, axis=1)))
                    output_slice = input_slice * updates
                elif reduce_op == "mean":
                    counts = tf.scatter_nd(tf.expand_dims(index_slice, axis=1), tf.ones_like(source_slice, dtype=tf.float32), tf.shape(input_slice))
                    updates = tf.scatter_nd(tf.expand_dims(index_slice, axis=1), source_slice * alpha, tf.shape(input_slice))
                    output_slice = input_slice + updates / counts
                elif reduce_op == "amax":
                    updates = tf.tensor_scatter_nd_update(input_slice, tf.expand_dims(index_slice, axis=1), tf.math.maximum(tf.gather_nd(source_slice * alpha, tf.expand_dims(index_slice, axis=1)), tf.gather_nd(input_slice, tf.expand_dims(index_slice, axis=1))))
                    output_slice = updates
                elif reduce_op == "amin":
                     updates = tf.tensor_scatter_nd_update(input_slice, tf.expand_dims(index_slice, axis=1), tf.math.minimum(tf.gather_nd(source_slice * alpha, tf.expand_dims(index_slice, axis=1)), tf.gather_nd(input_slice, tf.expand_dims(index_slice, axis=1))))
                     output_slice = updates
                else:
                    raise ValueError(f"Reduce op {reduce_op} is not supported.")
                output_inner_list.append(output_slice)
            output_list.append(tf.stack(output_inner_list, axis=0))
        
        output_stacked = tf.stack(output_list, axis=0)
        result = tf.reshape(output_stacked, tf.shape(input_tensor))

        result = result.numpy()
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "index": np.array([0, 1], dtype=np.int64),
        "source": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        "reduce": "sum",
        "dim": 0,
        "alpha": 1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()