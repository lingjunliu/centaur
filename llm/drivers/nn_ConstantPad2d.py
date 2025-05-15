import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict["padding"]
    value = input_dict["value"]
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    m = nn.ConstantPad2d(padding, value)
    result = m(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        padding = input_dict["padding"]
        value = input_dict["value"]

        input_shape = input_tensor.shape.as_list()
        if len(input_shape) == 2:
          hin, win = input_shape
          n = 1
          c = 1
        elif len(input_shape) == 3:
          c, hin, win = input_shape
          n = 1
        elif len(input_shape) == 4:
          n, c, hin, win = input_shape
        else:
          raise ValueError("Input tensor must have 2, 3 or 4 dimensions")
          
        if isinstance(padding, int):
          padding_left = padding_right = padding_top = padding_bottom = padding
        else:
          padding_left, padding_right, padding_top, padding_bottom = padding
        
        if len(input_shape) == 4:
            paddings = [[0, 0], [0, 0], [padding_top, padding_bottom], [padding_left, padding_right]]
        elif len(input_shape) == 3:
            paddings = [[0, 0], [padding_top, padding_bottom], [padding_left, padding_right]]
        else:
            paddings = [[padding_top, padding_bottom], [padding_left, padding_right]]
        
        result = tf.pad(input_tensor, paddings, "CONSTANT", constant_values=value)
    
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.randn(1, 2, 2).astype(np.float32),
        "padding": 2,
        "value": 3.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.randn(1, 2, 2).astype(np.float32),
        "padding": (3, 0, 2, 1),
        "value": 3.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.randn(2, 2).astype(np.float32),
        "padding": (3, 0, 2, 1),
        "value": 3.5
    }

    torch_result = torch_version(input_data)
    
    input_data['input'] = np.expand_dims(input_data['input'], axis=0)
    input_data['input'] = np.expand_dims(input_data['input'], axis=0)

    tf_result = tensorflow_version(input_data)

    torch_result['result'] = np.expand_dims(torch_result['result'], axis=0)
    torch_result['result'] = np.expand_dims(torch_result['result'], axis=0)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()