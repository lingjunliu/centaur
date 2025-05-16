import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    indices = torch.tensor(input_dict["indices"])
    output_size = input_dict.get("output_size", None)
    kernel_size = input_dict.get("kernel_size", (2, 2))
    stride = input_dict.get("stride", None)
    padding = input_dict.get("padding", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        indices = indices.cuda()

    max_unpool = torch.nn.MaxUnpool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    result = max_unpool(input_tensor, indices, output_size=output_size)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    indices = tf.constant(input_dict["indices"], dtype=tf.int32)
    output_size_np = input_dict.get("output_size", None)
    kernel_size = input_dict.get("kernel_size", (2, 2))
    stride = input_dict.get("stride", None)
    padding = input_dict.get("padding", 0)

    input_shape = tf.shape(input_tensor)
    batch_size = input_shape[0]
    height = input_shape[1]
    width = input_shape[2]
    channels = input_shape[3]

    if stride is None:
        stride_h = kernel_size[0]
        stride_w = kernel_size[1]
    elif isinstance(stride, int):
        stride_h = stride
        stride_w = stride
    elif isinstance(stride, tuple):
        stride_h = stride[0]
        stride_w = stride[1]
    else:
        raise ValueError("Invalid stride type")

    if output_size_np is None:
        output_height = (height - 1) * stride_h + kernel_size[0] - 2 * padding
        output_width = (width - 1) * stride_w + kernel_size[1] - 2 * padding
        output_shape_list = [batch_size, output_height, output_width, channels]
        output_shape = tf.constant([batch_size, output_height, output_width, channels], dtype=tf.int32)
    else:
        output_shape_list = [1, output_size_np[0], output_size_np[1], 1]
        output_shape = tf.constant(output_size_np, dtype=tf.int32)
        
    output_h = output_shape_list[1]
    output_w = output_shape_list[2]

    updates = tf.reshape(input_tensor, [-1])
    indices_reshaped = tf.reshape(indices, [-1])
    
    row_indices = indices_reshaped // width
    col_indices = indices_reshaped % width

    linear_indices = row_indices * output_w + col_indices

    sparse_tensor = tf.sparse.SparseTensor(indices=tf.expand_dims(tf.cast(linear_indices, tf.int64), axis=1), values=updates, dense_shape=[output_h * output_w])
    dense_tensor = tf.sparse.to_dense(sparse_tensor)
    result = tf.reshape(dense_tensor, [1, output_h, output_w, 1])

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[[1.0]]]], dtype=np.float32),
        "indices": np.array([[[[0]]]], dtype=np.int64),
        "kernel_size": (2, 2),
        "stride": (1, 1),
        "padding": 0,
        "output_size": (2, 2)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()