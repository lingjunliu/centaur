import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    k = input_dict["k"]
    group_size = input_dict.get("group_size", 1)
    pad_value = input_dict.get("pad_value", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    input_reshaped = input_tensor.reshape(-1, group_size)
    values, indices = torch.topk(torch.abs(input_reshaped), k=k, dim=-1)
    mask = torch.zeros_like(input_reshaped, dtype=torch.bool)
    mask.scatter_(dim=-1, index=indices, value=True)
    result_reshaped = torch.where(mask, input_reshaped, torch.tensor(pad_value, dtype=input_reshaped.dtype))
    result = result_reshaped.reshape(input_tensor.shape)

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
        k = input_dict["k"]
        group_size = input_dict.get("group_size", 1)
        pad_value = input_dict.get("pad_value", 0)

        input_shape = tf.shape(input_tensor)
        input_reshaped = tf.reshape(input_tensor, [-1, group_size])

        _, indices = tf.math.top_k(tf.abs(input_reshaped), k=k)

        mask = tf.zeros_like(input_reshaped, dtype=tf.bool)
        
        row_indices = tf.expand_dims(tf.range(tf.shape(indices)[0]), axis=1)
        
        full_indices = tf.stack([row_indices, indices], axis=-1)
        
        updates = tf.ones(tf.shape(indices), dtype=tf.bool)
        
        mask = tf.tensor_scatter_nd_update(mask, full_indices, updates)

        result_reshaped = tf.where(mask, input_reshaped, tf.cast(pad_value, input_reshaped.dtype))
        
        result = tf.reshape(result_reshaped, input_shape)
        
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0], dtype=np.float32),
        "k": 1,
        "group_size": 2,
        "pad_value": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()