import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    unflatten_dim = input_dict.get("dim", 0)
    unflatten_shape = tuple(input_dict["unflattened_size"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    unflatten = torch.nn.Unflatten(dim=unflatten_dim, unflattened_size=unflatten_shape)
    result = unflatten(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    unflatten_dim = input_dict.get("dim", 0)
    unflatten_shape = tuple(input_dict["unflattened_size"])

    input_shape = tf.shape(input_tensor)
    rank = tf.rank(input_tensor)

    if unflatten_dim < 0:
        unflatten_dim = unflatten_dim + rank

    prefix_shape = input_shape[:unflatten_dim]
    suffix_shape = input_shape[unflatten_dim + 1:]

    new_shape = tf.concat([tf.cast(prefix_shape, dtype=tf.int32), tf.constant(unflatten_shape, dtype=tf.int32), tf.cast(suffix_shape, dtype=tf.int32)], axis=0)
    
    result = tf.reshape(input_tensor, new_shape)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
        "dim": 0,
        "unflattened_size": [2, 3]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()