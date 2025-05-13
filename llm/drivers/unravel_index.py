import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    indices = torch.tensor(input_dict["indices"])
    shape = input_dict["shape"]
    
    if not cpu:
        indices = indices.cuda()
    
    result = torch.unravel_index(indices, shape)
    
    if not cpu:
        result = tuple(r.cpu() for r in result)
    
    return {"result": tuple(r.numpy() for r in result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    indices = tf.constant(input_dict["indices"])
    shape = input_dict["shape"]
    
    num_indices = tf.size(indices)
    unraveled_results = []
    
    for i in range(len(shape)):
        coordinate = tf.TensorArray(tf.int32, size=num_indices, dynamic_size=False)
        unraveled_results.append(coordinate)

    remainder = indices
    multiplier = 1

    for i in reversed(range(len(shape))):
        dimension = shape[i]
        coordinate = tf.math.floordiv(remainder, multiplier) % dimension
        unraveled_results[i] = unraveled_results[i].unstack(tf.cast(coordinate, tf.int32))
        multiplier *= dimension
    
    result = tuple(r.stack().numpy() for r in unraveled_results)
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "indices": np.array([22, 41, 37], dtype=np.int64),
        "shape": (7, 6, 7)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for i in range(len(torch_result['result'])):
        assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()