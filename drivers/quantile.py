import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    q = input_dict["q"]
    if isinstance(q, float):
        pass
    else:
        q = torch.tensor(q)
    dim = input_dict.get("dim", None)
    keepdim = input_dict.get("keepdim", False)
    interpolation = input_dict.get("interpolation", 'linear')

    if not cpu:
        input_tensor = input_tensor.cuda()
        if isinstance(q, torch.Tensor):
             q = q.cuda()

    result = torch.quantile(input_tensor, q, dim=dim, keepdim=keepdim, interpolation=interpolation)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    q = input_dict["q"]
    dim = input_dict.get("dim", None)
    keepdim = input_dict.get("keepdim", False)
    interpolation = input_dict.get("interpolation", 'linear')

    def quantile(tensor, q, axis=None, interpolation='linear'):
        if axis is not None:
            values = tf.sort(tensor, axis=axis)
            n = tf.cast(tf.shape(values)[axis], dtype=tf.float32)
        else:
            values = tf.sort(tf.reshape(tensor, [-1]))
            n = tf.cast(tf.shape(values)[0], dtype=tf.float32)
        
        q_index = q * (n - 1)
        
        if interpolation == 'linear':
            i = tf.floor(q_index)
            j = tf.math.ceil(q_index)
            i = tf.cast(i, dtype=tf.int32)
            j = tf.cast(j, dtype=tf.int32)
            
            if axis is not None:
                i_val = tf.gather(values, i, axis=axis)
                j_val = tf.gather(values, j, axis=axis)
            else:
                i_val = tf.gather(values, i)
                j_val = tf.gather(values, j)

            fraction = q_index - tf.floor(q_index)
            
            result = i_val + (j_val - i_val) * fraction
        elif interpolation == 'lower':
            i = tf.floor(q_index)
            i = tf.cast(i, dtype=tf.int32)
            if axis is not None:
                result = tf.gather(values, i, axis=axis)
            else:
                result = tf.gather(values, i)
        elif interpolation == 'higher':
            i = tf.math.ceil(q_index)
            i = tf.cast(i, dtype=tf.int32)
            if axis is not None:
                result = tf.gather(values, i, axis=axis)
            else:
                result = tf.gather(values, i)
        elif interpolation == 'nearest':
            i = tf.round(q_index)
            i = tf.cast(i, dtype=tf.int32)
            if axis is not None:
                result = tf.gather(values, i, axis=axis)
            else:
                result = tf.gather(values, i)
        elif interpolation == 'midpoint':
            i = tf.floor(q_index)
            j = tf.math.ceil(q_index)
            i = tf.cast(i, dtype=tf.int32)
            j = tf.cast(j, dtype=tf.int32)
            if axis is not None:
                i_val = tf.gather(values, i, axis=axis)
                j_val = tf.gather(values, j, axis=axis)
            else:
                i_val = tf.gather(values, i)
                j_val = tf.gather(values, j)

            result = (i_val + j_val) / 2.0
        else:
            raise ValueError("Invalid interpolation method")

        return result

    if isinstance(q, float):
        result = quantile(input_tensor, q, axis=dim, interpolation=interpolation)
    else:
        results = []
        for qi in q:
            results.append(quantile(input_tensor, qi, axis=dim, interpolation=interpolation))
        result = tf.stack(results)

        if dim == 1 and keepdim:
            result = tf.expand_dims(result, axis=1)
        elif dim == 0 and len(input_tensor.shape) > 1:
            perm = list(range(1, len(input_tensor.shape)))
            perm.insert(1, 0)
            result = tf.transpose(result, perm=perm)
            

    if keepdim and dim is not None:
        shape = list(input_tensor.shape)
        shape[dim] = 1
        if isinstance(q, float):
            result = tf.reshape(result, shape[:dim] + [1] + shape[dim+1:])
        else:
            if dim == 1:
                result = tf.reshape(result, [len(q)] + shape[:dim] + [1] + shape[dim+1:])
            else:
                 result = tf.reshape(result, [len(q)] + [1] + shape[1:])
    
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.0795, -1.2117, 0.9765], [1.1707, 0.6706, 0.4884]], dtype=np.float32),
        "q": np.array([0.25, 0.5, 0.75], dtype=np.float32),
        "dim": 1,
        "keepdim": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.arange(4., dtype=np.float32),
        "q": 0.6,
        "interpolation": 'linear'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()