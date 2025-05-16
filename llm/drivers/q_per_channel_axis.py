import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    scale = torch.tensor(input_dict["scale"])
    zero_point = torch.tensor(input_dict["zero_point"])
    axis = input_dict.get("axis", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        scale = scale.cuda()
        zero_point = zero_point.cuda()

    result = torch.quantize_per_channel(input_tensor, scale, zero_point, axis=axis, dtype=torch.quint8)
    result = result.dequantize()

    if not cpu:
        result = result.cpu()

    return {'result': result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        scale = tf.constant(input_dict["scale"], dtype=tf.float32)
        zero_point = tf.constant(input_dict["zero_point"], dtype=tf.float32)
        axis = input_dict.get("axis", 0)
        
        rank = len(input_tensor.shape)
        scale_shape = tf.shape(scale)[0]

        broadcast_shape = tf.one_hot(axis, depth=rank, on_value=scale_shape, off_value=1)
        broadcast_shape = tf.cast(broadcast_shape, dtype=tf.int32)
        broadcast_shape = tf.where(tf.equal(broadcast_shape, 1), tf.shape(input_tensor), broadcast_shape)
        
        scale = tf.reshape(scale, tf.shape(scale))
        zero_point = tf.reshape(zero_point, tf.shape(zero_point))
        
        expanded_scale = tf.reshape(scale, [1 if i != axis else -1 for i in range(rank)])
        expanded_zero_point = tf.reshape(zero_point, [1 if i != axis else -1 for i in range(rank)])
        

        result = (input_tensor - expanded_zero_point) * expanded_scale

        result = result.numpy()
    return {'result': result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        "scale": np.array([0.5, 0.25, 0.125], dtype=np.float32),
        "zero_point": np.array([0, 1, 2], dtype=np.float32),
        "axis": 1
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()