import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    indices = torch.tensor(input_dict["indices"])
    update = torch.tensor(input_dict["update"])
    accumulate = input_dict.get("accumulate", False)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        indices = indices.cuda()
        update = update.cuda()
    
    result = torch.ops.aten.index_put(input_tensor, [indices], update, accumulate=accumulate)
    
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
        input_tensor = tf.constant(input_dict["input"])
        indices = tf.constant(input_dict["indices"])
        update = tf.constant(input_dict["update"])
        accumulate = input_dict.get("accumulate", False)

        if accumulate:
            result = tf.tensor_scatter_nd_add(input_tensor, tf.expand_dims(indices, axis=1), update)
        else:
            result = tf.tensor_scatter_nd_update(input_tensor, tf.expand_dims(indices, axis=1), update)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "indices": np.array([0, 2], dtype=np.int32),
        "update": np.array([5.0, 6.0], dtype=np.float32),
        "accumulate": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "indices": np.array([0, 2], dtype=np.int32),
        "update": np.array([5.0, 6.0], dtype=np.float32),
        "accumulate": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()