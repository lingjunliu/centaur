import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input = torch.tensor(input_dict['input'])
    min_val = torch.tensor(input_dict['min'])
    max_val = torch.tensor(input_dict['max'])

    if not cpu:
        input = input.cuda()
        min_val = min_val.cuda()
        max_val = max_val.cuda()

    result = torch.clip(input, min=min_val, max=max_val)

    if not cpu:
        result = result.cpu()

    return {'result': result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input = tf.constant(input_dict['input'])
        min_val = tf.constant(input_dict['min'])
        max_val = tf.constant(input_dict['max'])

        result = tf.clip_by_value(input, clip_value_min=min_val, clip_value_max=max_val)

        result = result.numpy()

    return {'result': result}

def main():
    A_TOL = 0.01

    input_data = {
        'input': np.array([-1.0, 0.5, 2.0, 3.5], dtype=np.float32),
        'min': np.array([0.0], dtype=np.float32),
        'max': np.array([3.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()