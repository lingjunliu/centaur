import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    unflatten_dim = input_dict.get("dim", 0)
    unflatten_unflattened_size = tuple(input_dict["unflattened_size"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    unflatten = torch.nn.Unflatten(dim=unflatten_dim, unflattened_size=unflatten_unflattened_size)
    result = unflatten(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        unflatten_dim = input_dict.get("dim", 0)
        unflatten_unflattened_size = input_dict["unflattened_size"]

        input_shape = tf.shape(input_tensor)
        input_rank = len(input_shape.numpy())

        if unflatten_dim < 0:
            unflatten_dim = input_rank + unflatten_dim

        prefix_shape = input_shape[:unflatten_dim]
        suffix_shape = input_shape[unflatten_dim + 1:]

        prefix_shape_tf = prefix_shape
        suffix_shape_tf = suffix_shape
        unflattened_size_tf = tf.constant(unflatten_unflattened_size, dtype=tf.int32)

        new_shape = tf.concat([tf.cast(prefix_shape_tf, dtype=tf.int32), unflattened_size_tf, tf.cast(suffix_shape_tf, dtype=tf.int32)], axis=0)

        result = tf.reshape(input_tensor, new_shape)

        result = result.numpy()

    return {"result": result}

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
    
    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32).reshape(1,6),
        "dim": 1,
        "unflattened_size": [2, 3]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3, 4], dtype=np.float32).reshape(2, 2),
        "dim": 1,
        "unflattened_size": [4]
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()