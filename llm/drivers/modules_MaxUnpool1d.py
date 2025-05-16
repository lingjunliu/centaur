import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    indices = torch.tensor(input_dict["indices"])
    output_size = input_dict.get("output_size", None)
    stride = input_dict.get("stride", None)
    padding = input_dict.get("padding", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        indices = indices.cuda()

    maxunpool1d = torch.nn.MaxUnpool1d(kernel_size=1, stride=stride, padding=padding)

    if output_size is not None:
        result = maxunpool1d(input_tensor.unsqueeze(0).unsqueeze(0), indices.unsqueeze(0).unsqueeze(0), output_size=(1, output_size))
    else:
        result = maxunpool1d(input_tensor.unsqueeze(0).unsqueeze(0), indices.unsqueeze(0).unsqueeze(0))

    if not cpu:
        result = result.cpu()

    return {"result": result.squeeze().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor_np = input_dict["input"]
    indices_np = input_dict["indices"]
    output_size = input_dict.get("output_size", None)
    stride = input_dict.get("stride", None)
    padding = input_dict.get("padding", 0)

    input_tensor = tf.constant(input_tensor_np, dtype=tf.float32)
    indices = tf.constant(indices_np, dtype=tf.int64)

    input_tensor_expanded = tf.expand_dims(tf.expand_dims(input_tensor, axis=0), axis=0)
    indices_expanded = tf.expand_dims(tf.expand_dims(indices, axis=0), axis=0)

    input_shape = input_tensor_np.shape[0]
    if stride is None:
        stride = 1
    
    if output_size is None:
      output_size = (input_shape - 1) * stride + 1 + 2 * padding

    output_shape = [1, 1, output_size]

    updates = tf.reshape(input_tensor_expanded, [-1])
    indices_tf = tf.reshape(indices_expanded, [-1, 1])
    shape = tf.constant(output_shape, dtype=tf.int64)

    sparse_output = tf.scatter_nd(indices_tf, updates, shape)
    result = tf.reshape(sparse_output, output_shape)
    
    return {"result": result.numpy().squeeze()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "indices": np.array([2, 5, 8], dtype=np.int64),
        "stride": 3,
        "padding": 0,
        "output_size": 10
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([1.0, 2.0], dtype=np.float32),
        "indices": np.array([1, 3], dtype=np.int64),
        "output_size": 5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "indices": np.array([2, 5, 8], dtype=np.int64),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()