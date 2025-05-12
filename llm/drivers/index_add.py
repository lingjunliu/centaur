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
        input_tensor = tf.constant(input_dict["input"])
        index = tf.constant(input_dict["index"])
        other = tf.constant(input_dict["other"])
        dim = input_dict.get("dim", 0)
        alpha = input_dict.get("alpha", 1.0)

        index = tf.cast(index, dtype=tf.int64)

        if dim == 0:
          index_updates = tf.stack([index, tf.zeros_like(index, dtype=tf.int64)], axis=1)
        else:
          index_updates = tf.stack([tf.zeros_like(index, dtype=tf.int64), index], axis=1)

        updates = tf.tensor_scatter_nd_update(
            input_tensor,
            index_updates,
            alpha * other
        )
        
        result = updates
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        "index": np.array([0, 2, 1], dtype=np.int64),
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