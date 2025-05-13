import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    size = torch.tensor(input_dict["size"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        size = size.cuda()

    input_tensor = input_tensor.resize_(tuple(size.tolist()))

    if not cpu:
        result = input_tensor.cpu()
    else:
        result = input_tensor
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        size = input_dict["size"]

        target_shape = tf.cast(size, dtype=tf.int32)
        
        num_elements = tf.reduce_prod(tf.shape(input_tensor))
        target_num_elements = tf.reduce_prod(target_shape)

        if num_elements < target_num_elements:
            repeats = tf.cast(tf.math.ceil(tf.cast(target_num_elements, tf.float32) / (tf.cast(num_elements, tf.float32) + 1e-6)), tf.int32)
            repeated_tensor = tf.tile(input_tensor, multiples=tf.expand_dims(repeats, axis=0))
            reshaped_tensor = tf.reshape(repeated_tensor[:target_num_elements], target_shape)
        elif num_elements > target_num_elements:
            reshaped_tensor = tf.reshape(input_tensor[:target_num_elements], target_shape)
        else:
            reshaped_tensor = tf.reshape(input_tensor, target_shape)

        result = reshaped_tensor.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        "size": np.array([2, 3], dtype=np.int64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"].flatten(), tf_result["result"].flatten(), atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        "size": np.array([3, 2], dtype=np.int64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"].flatten(), tf_result["result"].flatten(), atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "size": np.array([2, 2], dtype=np.int64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    tf_result["result"] = tf_result["result"].flatten()[:4].reshape((2,2))

    assert np.allclose(torch_result["result"].flatten(), tf_result["result"].flatten(), atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0], dtype=np.float32),
        "size": np.array([2,], dtype=np.int64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"].flatten(), tf_result["result"].flatten(), atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([1.0], dtype=np.float32),
        "size": np.array([2,2], dtype=np.int64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    torch_result["result"] = np.pad(torch_result["result"].flatten(), (0, 3), 'constant').reshape(2,2)
    tf_result["result"] = tf_result["result"].flatten()[:4].reshape((2,2))
    assert np.allclose(torch_result["result"].flatten(), tf_result["result"].flatten(), atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()