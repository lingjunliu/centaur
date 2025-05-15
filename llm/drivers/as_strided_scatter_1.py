import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    values = torch.tensor(input_dict["values"])
    indices = torch.tensor(input_dict["indices"])
    
    shape = tuple(input_tensor.shape)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        values = values.cuda()
        indices = indices.cuda()

    result = torch.as_strided_scatter(input_tensor, values, indices, strided_size=shape)

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
        values = tf.constant(input_dict["values"])
        indices = input_dict["indices"]

        updates = tf.reshape(values, [-1])
        indices_tensor = tf.constant(indices, dtype=tf.int32)
        
        result = tf.tensor_scatter_nd_update(input_tensor, indices_tensor, updates)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.zeros((5, 5), dtype=np.float32),
        "values": np.array([1, 2, 3], dtype=np.float32),
        "indices": np.array([[1, 1], [2, 3], [4, 0]], dtype=np.int64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()