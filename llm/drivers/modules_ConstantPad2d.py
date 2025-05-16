import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

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
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    padding = input_dict["padding"]
    value = input_dict.get("value", 0.0)

    shape = input_tensor.shape
    if len(shape) == 2:
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        input_tensor = tf.expand_dims(input_tensor, axis=0)
    elif len(shape) == 3:
        input_tensor = tf.expand_dims(input_tensor, axis=0)
    
    if isinstance(padding, int):
        padding = [[0, 0], [padding, padding], [padding, padding], [0, 0]]
    elif isinstance(padding, tuple) and len(padding) == 2:
        padding = [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]]
    elif isinstance(padding, tuple) and len(padding) == 4:
        padding = [[0, 0], [padding[0], padding[1]], [padding[2], padding[3]], [0, 0]]

    result = tf.pad(input_tensor, padding, constant_values=value)

    if len(shape) == 2:
        result = tf.squeeze(result, axis=0)
        result = tf.squeeze(result, axis=0)
    elif len(shape) == 3:
        result = tf.squeeze(result, axis=0)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(3, 4, 5).astype(np.float32),
        "padding": (1, 2, 3, 4),
        "value": 0.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(4, 5).astype(np.float32),
        "padding": 2,
        "value": 0.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(4, 5).astype(np.float32),
        "padding": (1, 2),
        "value": 0.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()