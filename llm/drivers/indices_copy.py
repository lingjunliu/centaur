import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    index = torch.tensor(input_dict["index"]).long()
    source = torch.tensor(input_dict["source"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        index = index.cuda()
        source = source.cuda()
    
    input_tensor = input_tensor.clone()
    input_tensor[index] = source
    result = input_tensor
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.convert_to_tensor(input_dict["input"], dtype=tf.float32)
        index = tf.convert_to_tensor(input_dict["index"], dtype=tf.int32)
        source = tf.convert_to_tensor(input_dict["source"], dtype=tf.float32)

        rank = tf.rank(input_tensor)

        if rank == 1:
            indices = tf.expand_dims(index, axis=1)
            updates = source
        elif rank == 2:
            indices = tf.stack([index[:, 0], index[:, 1]], axis=1)
            updates = source
        else:
            raise ValueError("Only supports rank 1 and rank 2 tensors")

        result = tf.tensor_scatter_nd_update(tf.identity(input_tensor), indices, updates)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3, 4, 5], dtype=np.float32),
        "index": np.array([0, 2], dtype=np.int64),
        "source": np.array([10, 20], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        "index": np.array([[0, 0], [2, 1]], dtype=np.int64),
        "source": np.array([10, 20], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()