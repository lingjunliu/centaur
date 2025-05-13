import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    index = torch.tensor(input_dict["index"])
    source = torch.tensor(input_dict["source"])
    accumulate = input_dict.get("accumulate", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        index = index.cuda()
        source = source.cuda()

    result = torch.put(input_tensor, index, source, accumulate=accumulate)

    if not cpu:
        result = result.cpu()

    return {'result': result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor_np = input_dict["input"]
    index_np = input_dict["index"]
    source_np = input_dict["source"]
    accumulate = input_dict.get("accumulate", False)

    input_tensor = tf.Variable(input_tensor_np)
    index = tf.convert_to_tensor(index_np, dtype=tf.int32)
    source = tf.convert_to_tensor(source_np)
    
    if accumulate:
        updates = tf.tensor_scatter_nd_add(input_tensor, tf.expand_dims(index, axis=1), source)
    else:
        updates = tf.tensor_scatter_nd_update(input_tensor, tf.expand_dims(index, axis=1), source)

    result = updates.numpy()
    return {'result': result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "index": np.array([0, 2], dtype=np.int64),
        "source": np.array([10.0, 20.0], dtype=np.float32),
        "accumulate": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "index": np.array([0, 2], dtype=np.int64),
        "source": np.array([10.0, 20.0], dtype=np.float32),
        "accumulate": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()