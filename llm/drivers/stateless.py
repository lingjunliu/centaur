import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    weight = torch.tensor(input_dict['weight'])
    bias = torch.tensor(input_dict['bias'])
    named_params = {'weight': weight, 'bias': bias}

    if not cpu:
        named_params = {k: v.cuda() for k, v in named_params.items()}

    result = [named_params[name] for name in input_dict['param_names']]

    if not cpu:
        result = [p.cpu() for p in result]

    return {'result': [p.numpy() for p in result]}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    weight = tf.constant(input_dict['weight'])
    bias = tf.constant(input_dict['bias'])
    named_params = {'weight': weight, 'bias': bias}

    result = list(named_params[name].numpy() for name in input_dict['param_names'])

    return {'result': result}

def main():
    A_TOL = 0.01

    input_data = {
        'weight': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        'bias': np.array([0.5, 1.5], dtype=np.float32),
        'param_names': ['weight', 'bias']
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for torch_res, tf_res in zip(torch_result['result'], tf_result['result']):
        assert np.allclose(torch_res, tf_res, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()