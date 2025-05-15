import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    index = torch.tensor(input_dict["index"])
    other = torch.tensor(input_dict["other"])
    dim = input_dict.get("dim", 0)
    alpha = input_dict.get("alpha", 1.0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        index = index.cuda()
        other = other.cuda()

    result = torch.index_add(input_tensor, dim, index, other, alpha=alpha)

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
        input_tensor = tf.convert_to_tensor(input_dict["input"])
        index = tf.convert_to_tensor(input_dict["index"], dtype=tf.int32)
        other = tf.convert_to_tensor(input_dict["other"])
        dim = input_dict.get("dim", 0)
        alpha = input_dict.get("alpha", 1.0)

        input_shape = tf.shape(input_tensor)
        num_updates = tf.shape(index)[0]
        updates_shape = tf.concat([[num_updates], input_shape[1:]], axis=0)
        updates = tf.reshape(alpha * other, updates_shape)

        indices = tf.expand_dims(index, axis=1)
        rank = len(input_tensor.shape)
        if rank > 1:
            updates_indices = [indices]
            for i in range(1, rank):
                updates_indices.append(tf.expand_dims(tf.range(input_shape[i]), axis=0) + tf.zeros(shape=(1,1), dtype=tf.int32))
            updates_indices = tf.cast(tf.reshape(tf.stack(tf.meshgrid(*updates_indices, indexing='ij'), axis=-1), (-1, rank)), dtype=tf.int32)
            updates = tf.reshape(updates, (-1,))
            result = tf.tensor_scatter_nd_add(input_tensor, updates_indices, updates)
        else:
            result = tf.tensor_scatter_nd_add(input_tensor, tf.expand_dims(index, axis=1), updates)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        "index": np.array([0, 1, 0], dtype=np.int64),
        "other": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
        "dim": 0,
        "alpha": 0.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()