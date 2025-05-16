import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    data = input_dict["data"]
    dtype = input_dict.get("dtype", np.float32)
    
    if not cpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    storage = torch.FloatStorage.from_buffer(data, byte_order='native')

    result = torch.tensor(list(storage)).to(torch.float32).to(device)
    
    if not cpu:
        result = result.cpu()
        
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    data = input_dict["data"]
    dtype = input_dict.get("dtype", np.float32)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        numpy_array = np.frombuffer(data, dtype=dtype)

        tf_tensor = tf.constant(numpy_array, dtype=tf.as_dtype(dtype))

        result = tf_tensor.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    # Example input
    input_data = {
        "data": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32).tobytes(),
        "dtype": np.float32
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    # Assert to see if they are equal
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()