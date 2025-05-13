import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input = torch.tensor(input_dict['input'])
    dim = input_dict['dim']
    index = input_dict['index']
    value = torch.tensor(input_dict['value'])

    if not cpu:
        input = input.cuda()
        value = value.cuda()

    result = input.clone()
    
    index_tensor = torch.tensor(index)
    
    if not cpu:
        index_tensor = index_tensor.cuda()
        
    indices = index_tensor.tolist()

    if dim == 0:
        result[indices] = value
    elif dim == 1:
        result[:, indices] = value
    else:
        raise ValueError("Only dim 0 and 1 supported")

    if not cpu:
        result = result.cpu()

    return {'result': result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input = tf.constant(input_dict['input'])
    dim = input_dict['dim']
    index = input_dict['index']
    value = tf.constant(input_dict['value'])

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):

        if dim == 0:
          indices = tf.expand_dims(index, axis=1)
          update_indices = tf.cast(indices, dtype=tf.int64)
          updates = value
          result = tf.tensor_scatter_nd_update(input, update_indices, updates)
        elif dim == 1:
            num_rows = tf.shape(input)[0]
            num_cols = tf.shape(input)[1]
            
            row_indices = tf.range(num_rows, dtype=tf.int64)
            row_indices = tf.expand_dims(row_indices, axis=1)
            
            col_indices = tf.cast(index, dtype=tf.int64)
            
            row_indices_tiled = tf.tile(row_indices, [1, tf.shape(col_indices)[0]])

            col_indices_tiled = tf.tile(tf.expand_dims(col_indices, axis=0), [num_rows, 1])
            
            stacked_indices = tf.stack([row_indices_tiled, col_indices_tiled], axis=-1)
            
            update_indices = tf.reshape(stacked_indices, [-1, 2])
            updates = tf.reshape(value, [-1])
            result = tf.tensor_scatter_nd_update(input, update_indices, updates)
        else:
          raise ValueError("Only dim 0 and 1 are supported")
        
        result = result.numpy()

    return {'result': result}

def main():
    A_TOL = 0.01

    input_data = {
        'input': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        'dim': 0,
        'index': [0, 2],
        'value': np.array([[10, 11, 12], [13, 14, 15]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    input_data = {
        'input': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        'dim': 1,
        'index': [0, 2],
        'value': np.array([[10, 13], [11, 14], [12, 15]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()