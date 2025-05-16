import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    source = torch.tensor(input_dict["source"])
    diagonal = input_dict.get("diagonal", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        source = source.cuda()

    result = torch.diagonal_scatter(input_tensor, source, offset=diagonal)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    source = tf.constant(input_dict["source"])
    diagonal = input_dict.get("diagonal", 0)

    rank = len(input_tensor.shape)
    diag_len = source.shape[0]
    input_shape = tf.shape(input_tensor)

    if diagonal >= rank or diagonal < -rank:
      diagonal = max(min(diagonal, rank - 1), -rank + 1)
      

    if diagonal < 0:
        diagonal = diagonal + rank

    indices = tf.range(diag_len)
    diag_indices = []
    for i in range(rank):
      if i == diagonal:
        diag_indices.append(indices)
      else:
        diag_indices.append(tf.range(diag_len))

    diag_indices = tf.stack(diag_indices, axis=1)
    updates = tf.reshape(source, [diag_len])
    
    valid_indices = tf.reduce_all(diag_indices < tf.expand_dims(tf.cast(input_shape, tf.int32), 0), axis=1)
    diag_indices = tf.boolean_mask(diag_indices, valid_indices)
    updates = tf.boolean_mask(updates, valid_indices)

    result = tf.tensor_scatter_nd_update(input_tensor, tf.cast(diag_indices, tf.int32), updates)
    
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.zeros((5, 5), dtype=np.float32),
        "source": np.array([1, 2, 3, 4, 5], dtype=np.float32),
        "diagonal": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.zeros((5, 5), dtype=np.float32),
        "source": np.array([1, 2, 3], dtype=np.float32),
        "diagonal": 2
    }

    input_data["input"][0, 0] = 1
    input_data["input"][1, 1] = 1
    input_data["input"][2, 2] = 1
    input_data["input"][3, 3] = 1
    input_data["input"][4, 4] = 1

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    input_data = {
        "input": np.zeros((4, 4, 4), dtype=np.float32),
        "source": np.array([1, 2, 3, 4], dtype=np.float32),
        "diagonal": 1
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()