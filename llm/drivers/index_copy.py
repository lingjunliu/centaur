import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    index = torch.tensor(input_dict["index"])
    source = torch.tensor(input_dict["source"])
    dim = input_dict.get("dim", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        index = index.cuda()
        source = source.cuda()
    
    result = torch.index_copy(input_tensor, dim, index, source)
    
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
        input_tensor = tf.convert_to_tensor(input_dict["input"], dtype=tf.float32)
        index = tf.convert_to_tensor(input_dict["index"], dtype=tf.int32)
        source = tf.convert_to_tensor(input_dict["source"], dtype=tf.float32)
        dim = input_dict.get("dim", 0)

        input_shape = input_tensor.shape
        index_shape = index.shape
        source_shape = source.shape

        result = tf.identity(input_tensor)

        for i in range(index_shape[0]):
            idx = index[i]
            
            if dim == 0:
                updates = tf.expand_dims(source[i], axis=0)
                indices = [[idx]]

                result = tf.tensor_scatter_nd_update(result, indices, updates)
            elif dim == 1:

                indices_list = []
                for k in range(input_shape[0]):
                    indices_list.append([k, idx])
                
                indices = tf.convert_to_tensor(indices_list, dtype=tf.int32)
                
                updates = source[:, i] if len(source.shape) > 1 else source[i]

                result = tf.tensor_scatter_nd_update(result, indices, tf.reshape(updates, (input_shape[0],) if len(updates.shape)==0 else (input_shape[0],)))

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "index": np.array([0, 2], dtype=np.int64),
        "source": np.array([[10, 20, 30], [70, 80, 90]], dtype=np.float32),
        "dim": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data2 = {
        "input": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "index": np.array([0, 2], dtype=np.int64),
        "source": np.array([[10, 20, 30], [70, 80, 90]], dtype=np.float32)[:, :2],
        "dim": 1
    }
    
    input_data = {
        "input": np.array([[1, 2], [4, 5], [7, 8]], dtype=np.float32),
        "index": np.array([0, 1], dtype=np.int64),
        "source": np.array([[10, 20], [70, 80]], dtype=np.float32),
        "dim": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    

    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        "index": np.array([0, 1], dtype=np.int64),
        "source": np.array([[10, 20, 0], [70, 80, 0]], dtype=np.float32),
        "dim": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()