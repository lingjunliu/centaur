import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    if "gain" in input_dict:
        gain = input_dict["gain"]
    else:
        gain = 1.0

    if "nonlinearity" in input_dict:
        nonlinearity = input_dict["nonlinearity"]
    else:
        nonlinearity = 'linear'
    
    result = torch.nn.init.calculate_gain(nonlinearity, gain)
    
    if not cpu:
        result = torch.tensor(result).cpu().numpy()
    else:
        result = np.array(result)

    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if "gain" in input_dict:
        gain = input_dict["gain"]
    else:
        gain = 1.0

    if "nonlinearity" in input_dict:
        nonlinearity = input_dict["nonlinearity"]
    else:
        nonlinearity = 'linear'
    
    if nonlinearity == 'linear':
        result = gain
    elif nonlinearity == 'conv2d':
        result = gain
    elif nonlinearity == 'sigmoid':
        result = gain * (2/3)
    elif nonlinearity == 'tanh':
        result = gain * (5/3)
    elif nonlinearity == 'relu':
        result = gain * np.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if "negative_slope" in input_dict:
            negative_slope = input_dict["negative_slope"]
        else:
            negative_slope = 0.01
        result = gain * np.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
    
    return {"result": np.array(result)}

def torch_version_uniform(input_dict, cpu=True):
    import torch

    tensor = torch.tensor(input_dict["tensor"])

    if "a" in input_dict:
        a = input_dict["a"]
    else:
        a = 0.0
    
    if "b" in input_dict:
        b = input_dict["b"]
    else:
        b = 1.0
    
    if not cpu:
        tensor = tensor.cuda()
    
    torch.nn.init.uniform_(tensor, a=a, b=b)
    
    if not cpu:
        tensor = tensor.cpu()
    
    return {"result": tensor.numpy()}

def tensorflow_version_uniform(input_dict, cpu=True):
    import tensorflow as tf

    tensor = tf.constant(input_dict["tensor"])
    
    if "a" in input_dict:
        a = input_dict["a"]
    else:
        a = 0.0
    
    if "b" in input_dict:
        b = input_dict["b"]
    else:
        b = 1.0

    shape = tensor.shape
    
    result = tf.random.uniform(shape=shape, minval=a, maxval=b)
    result = tf.cast(result, dtype=tensor.dtype)
    
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "nonlinearity": 'relu',
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data_uniform = {
        "tensor": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "a": -1.0,
        "b": 1.0
    }
    torch_result_uniform = torch_version_uniform(input_data_uniform)
    tf_result_uniform = tensorflow_version_uniform(input_data_uniform)

    print("Success")

if __name__ == "__main__":
    main()