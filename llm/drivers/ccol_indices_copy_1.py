import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    indices = torch.tensor(input_dict["indices"])
    other = torch.tensor(input_dict["other"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        indices = indices.cuda()
        other = other.cuda()

    result = input_tensor.clone()
    for i, idx in enumerate(indices):
        result[:, idx] = other[:, i]

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    indices = tf.constant(input_dict["indices"])
    other = tf.constant(input_dict["other"])

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):

        result = tf.identity(input_tensor)
        for i in range(tf.shape(indices)[0]):
            index = indices[i]
            update = other[:, i]
            indices_to_update = tf.stack([tf.range(tf.shape(input_tensor)[0], dtype=tf.int64), tf.cast(tf.ones(tf.shape(input_tensor)[0], dtype=tf.int64) * index, dtype=tf.int64)], axis=1)
            result = tf.tensor_scatter_nd_update(result, indices_to_update, update)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "indices": np.array([0, 2], dtype=np.int64),
        "other": np.array([[7.0, 9.0], [8.0, 10.0]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()