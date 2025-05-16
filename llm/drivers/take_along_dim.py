import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    indices = torch.tensor(input_dict["indices"], dtype=torch.int64)
    dim = input_dict["dim"]

    if not cpu:
        input_tensor = input_tensor.cuda()
        indices = indices.cuda()

    result = torch.take_along_dim(input_tensor, indices, dim)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    indices = tf.constant(input_dict["indices"])
    dim = input_dict["dim"]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_shape = tf.shape(input_tensor)
        indices_shape = tf.shape(indices)
        rank = len(input_tensor.shape)

        if len(indices.shape) == 1:
            indices = tf.expand_dims(indices, axis=1)
            
        if dim < 0:
            dim = dim + rank

        ind_shape = tf.shape(indices)
        ind_rank = tf.shape(ind_shape)[0]
        inp_shape = tf.shape(input_tensor)
        
        
        def gather_along_dim(tensor, indices, dim):
            rank = len(tensor.shape)
            
            if dim < 0:
                dim = dim + rank

            idx_shape = tf.shape(indices)
            result_shape = tf.concat([tf.shape(tensor)[:dim], idx_shape, tf.shape(tensor)[dim+1:]], axis=0)
            
            
            if rank == 2:

                ind_range = tf.range(tf.shape(indices)[0])
                ind_2d = tf.stack([ind_range, tf.reshape(indices, [-1])], axis=1)

                result = tf.gather_nd(tensor, ind_2d)
                
                return result

            
            transposition = list(range(rank))
            transposition[0] = dim
            transposition[dim] = 0
            
            permuted_tensor = tf.transpose(tensor, perm=transposition)
            
            tensor_shape = tf.shape(tensor)
            reshaped_tensor = tf.reshape(permuted_tensor, (tensor_shape[dim], -1))
            
            gathered_values = tf.gather(reshaped_tensor, tf.reshape(indices, [-1]))

            result = tf.reshape(gathered_values, result_shape)
           
            return result

        result = gather_along_dim(input_tensor, indices, dim)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "indices": np.array([[0, 0], [1, 0]], dtype=np.int32),
        "dim": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "indices": np.array([[0, 1, 2], [0, 0, 0], [2, 2, 1]], dtype=np.int32),
        "dim": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32).reshape(2, 3),
        "indices": np.array([1, 2], dtype=np.int32).reshape(2, 1),
        "dim": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()