import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input = torch.tensor(input_dict['input'])
    mat2 = torch.tensor(input_dict['mat2'])

    if not cpu:
        input = input.cuda()
        mat2 = mat2.cuda()

    result = torch.sparse.mm(input, mat2)

    if not cpu:
        result = result.cpu()

    return {'result': result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    from tensorflow.python.ops import array_ops

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input = tf.constant(input_dict['input'])
        mat2 = tf.constant(input_dict['mat2'])

        input_sparse = tf.sparse.from_dense(input)
        mat2_sparse = tf.sparse.from_dense(mat2)

        result = tf.sparse.sparse_dense_matmul(input_sparse, mat2)
        
        result = result.numpy()

    return {'result': result}

def main():
    A_TOL = 0.01

    input_data = {
        'input': np.array([[0, 1, 0], [0, 0, 2], [1, 0, 0]], dtype=np.float32),
        'mat2': np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()