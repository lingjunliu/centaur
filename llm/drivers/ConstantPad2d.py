import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict["padding"]
    value = input_dict.get("value", 0.0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    pad = torch.nn.ConstantPad2d(padding, value)
    result = pad(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    padding = input_dict["padding"]
    value = input_dict.get("value", 0.0)

    input_shape = input_tensor.shape
    
    tf_padding = [[0, 0], [padding[0], padding[1]], [padding[2], padding[3]], [0, 0]]

    if len(input_shape) == 2:
        input_tensor = tf.expand_dims(tf.expand_dims(input_tensor, axis=0), axis=0)
        result = tf.pad(input_tensor, tf_padding, constant_values=value)
        result = tf.squeeze(result, axis=[0, 1])
    elif len(input_shape) == 3:
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        result = tf.pad(input_tensor, tf_padding, constant_values=value)
        result = tf.squeeze(result, axis=0)
    elif len(input_shape) == 4:
        result = tf.pad(input_tensor, tf_padding, constant_values=value)
    else: 
        raise ValueError("Input tensor must be 2D, 3D or 4D")

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "padding": (1, 1, 2, 0),
        "value": -1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(3, 5, 5).astype(np.float32),
        "padding": (2, 2, 2, 2),
        "value": 0.5
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.random.rand(5, 5).astype(np.float32),
        "padding": (2, 2, 2, 2),
        "value": 0.5
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.random.rand(1, 5, 5, 3).astype(np.float32),
        "padding": (2, 2, 2, 2),
        "value": 0.5
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.random.rand(2, 3, 4, 5).astype(np.float32),
        "padding": (1, 2, 3, 4),
        "value": 0.25
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    
    input_data = {
        "input": np.random.rand(1, 1, 4, 5).astype(np.float32),
        "padding": (1, 2, 3, 4),
        "value": 0.25
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()