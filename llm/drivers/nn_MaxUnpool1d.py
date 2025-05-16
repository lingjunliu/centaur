import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    indices = torch.tensor(input_dict["indices"])
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", None)
    padding = input_dict.get("padding", 0)
    output_size = input_dict.get("output_size", None)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        indices = indices.cuda()
    
    unpool = torch.nn.MaxUnpool1d(kernel_size, stride=stride, padding=padding)
    
    if not cpu:
        unpool = unpool.cuda()
    
    if output_size is not None:
        result = unpool(input_tensor, indices, output_size=output_size)
    else:
        result = unpool(input_tensor, indices)
        
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
        input_tensor = tf.convert_to_tensor(input_dict["input"], dtype=tf.float32)
        indices = tf.convert_to_tensor(input_dict["indices"], dtype=tf.int64)
        kernel_size = input_dict["kernel_size"]
        stride = input_dict.get("stride", None)
        padding = input_dict.get("padding", 0)
        output_size = input_dict.get("output_size", None)

        input_shape = tf.shape(input_tensor)
        N = input_shape[0]
        C = input_shape[1]
        H_in = input_shape[2]

        if stride is None:
            stride = kernel_size

        if output_size is None:
            H_out = (H_in - 1) * stride - 2 * padding + kernel_size
            output_size = [N, C, H_out]
        else:
            output_size = tf.TensorShape(output_size).as_list()

        updates = tf.reshape(input_tensor, [-1])
        indices_flat = tf.reshape(indices, [-1])
        shape = tf.reduce_prod(output_size)

        sparse_tensor = tf.sparse.SparseTensor(
            indices=tf.expand_dims(indices_flat, axis=1),
            values=updates,
            dense_shape=[shape]
        )

        dense_tensor = tf.sparse.to_dense(sparse_tensor)
        output_tensor = tf.reshape(dense_tensor, output_size)
    
    return {"result": output_tensor.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[1., 2]]], dtype=np.float32),
        "indices": np.array([[[0, 1]]], dtype=np.int64),
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "output_size": [1, 1, 2]
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[[1., 2, 3]]], dtype=np.float32),
        "indices": np.array([[[0, 1, 2]]], dtype=np.int64),
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "output_size": [1, 1, 3]
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()