import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input = torch.tensor(input_dict['input'])
    row_indices = torch.tensor(input_dict['row_indices'])
    col_indices = torch.tensor(input_dict['col_indices'])

    if not cpu:
        input = input.cuda()
        row_indices = row_indices.cuda()
        col_indices = col_indices.cuda()

    crow_indices = torch.stack([row_indices, col_indices], dim=0)
    result = torch.sparse_coo_tensor(crow_indices, input, size=(torch.max(row_indices).item() + 1, torch.max(col_indices).item() + 1)).to_dense()

    if not cpu:
        result = result.cpu()

    return {'result': result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_np = input_dict['input']
    row_indices_np = input_dict['row_indices']
    col_indices_np = input_dict['col_indices']

    input_tf = tf.convert_to_tensor(input_np)
    row_indices_tf = tf.convert_to_tensor(row_indices_np)
    col_indices_tf = tf.convert_to_tensor(col_indices_np)

    result_tf = tf.raw_ops.SparseToDense(
        sparse_indices=tf.stack([row_indices_tf, col_indices_tf], axis=1),
        sparse_values=input_tf,
        output_shape=[tf.reduce_max(row_indices_tf) + 1, tf.reduce_max(col_indices_tf) + 1],
        default_value=0)

    return {'result': result_tf.numpy()}

def main():
    A_TOL = 0.01
    input_data = {
        'input': np.array([1, 2, 3], dtype=np.float32),
        'row_indices': np.array([0, 1, 2], dtype=np.int64),
        'col_indices': np.array([0, 1, 0], dtype=np.int64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL)

    print("Success")

if __name__ == "__main__":
    main()