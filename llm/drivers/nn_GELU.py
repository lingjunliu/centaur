import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    input_tensor = torch.tensor(input_dict["input"])
    approximate = input_dict.get("approximate", 'none')

    if not cpu:
        input_tensor = input_tensor.cuda()

    m = torch.nn.GELU(approximate=approximate)
    result = m(input_tensor)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        approximate = input_dict.get("approximate", 'none')

        if approximate == 'none':
            result = 0.5 * input_tensor * (1.0 + tf.math.erf(input_tensor / tf.math.sqrt(2.0)))
        elif approximate == 'tanh':
            result = 0.5 * input_tensor * (1.0 + tf.math.tanh(tf.math.sqrt(2.0 / np.pi) * (input_tensor + 0.044715 * tf.pow(input_tensor, 3))))
        else:
            raise ValueError("Invalid approximate value")

        result = result.numpy()

    return {'result': result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "approximate": 'tanh'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "approximate": 'none'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()