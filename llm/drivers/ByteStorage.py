import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    storage = input_tensor.storage()
    
    if not cpu:
        storage = storage.cpu()

    result = np.array([storage[i] for i in range(len(storage))])
    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    
    if not cpu:
        device_string = "/GPU:0"
    else:
        device_string = "/CPU:0"

    with tf.device(device_string):
        tensor_bytes = tf.io.serialize_tensor(input_tensor)
        
        tensor_shape = input_tensor.shape
        tensor_size = tf.size(input_tensor)
        tensor_dtype = input_tensor.dtype
        
        flat_tensor = tf.reshape(input_tensor, [-1])
        
        return {"result": flat_tensor.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()