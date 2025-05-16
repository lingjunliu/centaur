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
    result = maxunpool1d(input_tensor.unsqueeze(0).unsqueeze(0), indices.unsqueeze(0).unsqueeze(0), output_size=output_size)
    result = result.squeeze()

    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    indices = tf.constant(input_dict["indices"], dtype=tf.int64)
    output_size = input_dict.get("output_size", None)
    stride = input_dict.get("stride", None)
    padding = input_dict.get("padding", 0)
    
    input_shape = input_tensor.shape
    indices_shape = indices.shape
    
    if stride is None:
        stride = 1
    
    if output_size is None:
        output_size_calculated = (input_shape[0] - 1) * stride + 1 - 2 * padding
        output_size = [output_size_calculated]
    
    batch_size = 1
    channels = 1
    input_length = input_shape[0]
    output_length = output_size[0]
    
    updates = tf.reshape(input_tensor, [batch_size * channels * input_length])
    indices_reshaped = tf.reshape(indices, [batch_size * channels * input_length])
    
    output_tensor = tf.scatter_nd(
        indices=tf.expand_dims(indices_reshaped, axis=1),
        updates=updates,
        shape=[batch_size * channels * output_length]
    )
    
    output_tensor = tf.reshape(output_tensor, [batch_size, channels, output_length])
    output_tensor = tf.squeeze(output_tensor)
    
    return {"result": output_tensor.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0], dtype=np.float32),
        "indices": np.array([0, 2], dtype=np.int64),
        "output_size": (1, 1, 4),
        "stride": 1,
        "padding": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "indices": np.array([0, 2, 5], dtype=np.int64),
        "output_size": (1, 1, 7),
        "stride": 1,
        "padding": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()