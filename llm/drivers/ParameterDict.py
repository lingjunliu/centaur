import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from torch.nn import ParameterDict, Parameter

    params = ParameterDict()
    for key, value in input_dict.items():
        params[key] = Parameter(torch.tensor(value))

    if not cpu:
        for key, param in params.items():
            params[key] = Parameter(param.cuda())

    result = {}
    for key, param in params.items():
        result[key] = param.detach()
        if not cpu:
            result[key] = result[key].cpu()
        result[key] = result[key].numpy()

    return {'result': result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        params = {}
        for key, value in input_dict.items():
            params[key] = tf.Variable(value)

        result = {}
        for key, param in params.items():
            result[key] = param.numpy()

    return {'result': result}

def main():
    A_TOL = 0.01

    input_data = {
        'param1': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'param2': np.array([4.0, 5.0, 6.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for key in input_data.keys():
        assert np.allclose(torch_result['result'][key], tf_result['result'][key], atol=A_TOL), f"Results for {key} do not match"

    print("Success")

if __name__ == "__main__":
    main()