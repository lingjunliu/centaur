import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    tensors = [torch.tensor(tensor) for tensor in input_dict['tensors']]
    axis = input_dict.get('axis', 0)

    if not cpu:
        tensors = [tensor.cuda() for tensor in tensors]

    result = torch.concatenate(tensors, axis=axis)

    if not cpu:
        result = result.cpu()

    return {'result': result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    tensors = [tf.constant(tensor) for tensor in input_dict['tensors']]
    axis = input_dict.get('axis', 0)

    result = tf.concat(tensors, axis=axis)

    return {'result': result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        'tensors': [
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            np.array([[5, 6]], dtype=np.float32)
        ],
        'axis': 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    input_data = {
        'tensors': [
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            np.array([[5, 6], [7,8]], dtype=np.float32)
        ],
        'axis': 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()