import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    params = torch.nn.ParameterDict()
    for key, value in input_dict.items():
        params[key] = torch.nn.Parameter(torch.tensor(value))

    if not cpu:
        for key in params:
            params[key] = torch.nn.Parameter(params[key].cuda())

    result = {}
    for key, param in params.items():
        if not cpu:
            result[key] = param.cpu().detach().numpy()
        else:
            result[key] = param.detach().numpy()

    return result

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

    return result

def main():
    A_TOL = 0.01
    input_data = {
        "param1": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "param2": np.array([4.0, 5.0, 6.0], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for key in input_data:
        assert np.allclose(torch_result[key], tf_result[key], atol=A_TOL), f"Results do not match for key: {key}"

    print("Success")

if __name__ == "__main__":
    main()