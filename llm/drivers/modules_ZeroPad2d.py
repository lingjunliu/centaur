import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    padding = input_dict["padding"]
    input_tensor = torch.tensor(input_dict["input"])
    value = input_dict.get("value", 0.0)

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.nn.functional.pad(input_tensor, padding, mode='constant', value=float(value))
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    padding = input_dict["padding"]
    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    value = float(input_dict.get("value", 0.0))
    
    rank = len(input_tensor.shape)
    
    if rank == 2:
      input_tensor = tf.expand_dims(input_tensor, axis=0)
      input_tensor = tf.expand_dims(input_tensor, axis=-1)

    elif rank == 3:
      input_tensor = tf.expand_dims(input_tensor, axis=0)

    if len(padding) == 2:
        pad_width = [[0, 0], [padding[0], padding[1]], [padding[0], padding[1]], [0, 0]]
    elif len(padding) == 4:
        pad_width = [[0, 0], [padding[0], padding[1]], [padding[2], padding[3]], [0, 0]]
    else:
        raise ValueError("Padding must be a tuple of length 2 or 4")

    result = tf.pad(input_tensor, pad_width, constant_values=value)
    
    if rank == 2:
        result = tf.squeeze(result, axis=[0, 3])
    elif rank == 3:
        result = tf.squeeze(result, axis=0)
    
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "padding": (1, 2, 0, 3),
        "value": 1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[[1, 2], [3, 4]]], dtype=np.float32),
        "padding": (1, 1),
        "value": 2.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "padding": (2, 2),
        "value": 3.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()