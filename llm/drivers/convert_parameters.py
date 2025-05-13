import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils
import tensorflow as tf

def torch_version(input_dict, cpu=True):
    parameters = input_dict["parameters"]
    if isinstance(parameters, list):
        parameters = [nn.Parameter(torch.tensor(p)) for p in parameters]
    else:
        parameters = [nn.Parameter(torch.tensor(parameters))]

    if not cpu:
        parameters = [p.cuda() for p in parameters]
    
    flat_param = utils.parameters_to_vector(parameters)

    if not cpu:
        flat_param = flat_param.cpu()
    
    return {"result": flat_param.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    parameters = input_dict["parameters"]
    if isinstance(parameters, list):
        parameter_vars = [tf.Variable(p, trainable=False) for p in parameters]
    else:
        parameter_vars = [tf.Variable(parameters, trainable=False)]

    parameter_concat = tf.concat([tf.reshape(v, [-1]) for v in parameter_vars], axis=0)

    return {"result": parameter_concat.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "parameters": [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), np.array([5.0, 6.0], dtype=np.float32)]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "parameters": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()