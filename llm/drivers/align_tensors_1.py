import numpy as np
import torch
import tensorflow as tf

def torch_version(input_dict, cpu=True):
    input_tensor = torch.tensor(input_dict["input"])
    alignment = torch.tensor(input_dict["alignment"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        alignment = alignment.cuda()

    input_shape = input_tensor.shape
    alignment_shape = alignment.shape

    rank_diff = max(0, len(alignment_shape) - len(input_shape))

    padding_shape = (rank_diff,) + input_shape

    padding = torch.zeros_like(alignment, dtype=input_tensor.dtype)

    slices = [slice(None) for _ in range(len(alignment_shape))]
    for i in range(rank_diff):
        slices[i] = 0
    
    slices = tuple(slices)

    try:
        padding[slices] = input_tensor
    except:
        padding = torch.zeros(alignment_shape, dtype=input_tensor.dtype)
        padding[slices] = input_tensor

    result = padding
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        alignment = tf.constant(input_dict["alignment"])

        input_shape = tf.shape(input_tensor)
        alignment_shape = tf.shape(alignment)

        rank_diff = tf.maximum(0, tf.rank(alignment) - tf.rank(input_tensor))

        padding = tf.zeros_like(alignment, dtype=input_tensor.dtype)
        
        if tf.rank(input_tensor) == 0:
            slices = [0] * tf.rank(alignment)
            indices = tf.expand_dims(tf.stack(slices), axis=0)
            updates = tf.reshape(input_tensor, [1])
        else:
            slices_list = []
            for i in range(tf.rank(alignment)):
                if i < rank_diff:
                    slices_list.append(0)
                else:
                    slices_list.append(tf.range(tf.shape(input_tensor)[i-rank_diff]))
            
            mesh = tf.meshgrid(*slices_list, indexing='ij')
            indices = tf.stack(mesh, axis=-1)
            
            if tf.rank(indices) > 1:
                indices = tf.reshape(indices, (-1, tf.rank(alignment)))
            else:
                indices = tf.expand_dims(indices, axis=0)
            updates = tf.reshape(input_tensor, [-1])
        
        if tf.rank(updates) == 0:
            updates = tf.expand_dims(updates, axis=0)
        
        padding = tf.tensor_scatter_nd_update(padding, indices, updates)

        result = padding.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "alignment": np.array([[1, 0], [1, 0]], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3], dtype=np.float32),
        "alignment": np.array([[0, 1, 0], [1, 0, 1]], dtype=np.float32)
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array(1, dtype=np.float32),
        "alignment": np.array([[1, 0], [1, 0]], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()