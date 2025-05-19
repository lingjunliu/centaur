import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    other_tensor = torch.tensor(input_dict["other"]) if isinstance(input_dict["other"], np.ndarray) else input_dict["other"]
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        if isinstance(other_tensor, torch.Tensor):
            other_tensor = other_tensor.cuda()
    
    result = torch.greater(input_tensor, other_tensor)
    
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
        other_tensor = tf.constant(input_dict["other"]) if isinstance(input_dict["other"], np.ndarray) else input_dict["other"]

        result = tf.greater(input_tensor, other_tensor)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data_array = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "other": np.array([0.01, 1.1, 1.3, -0.7], dtype=np.float32),
    }
    input_data_scalar = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "other": 1.0,
    }

    torch_result_array = torch_version(input_data_array)
    tf_result_array = tensorflow_version(input_data_array)
    assert np.allclose(torch_result_array["result"], tf_result_array["result"], atol=A_TOL), "Results do not match (array)"

    torch_result_scalar = torch_version(input_data_scalar)
    tf_result_scalar = tensorflow_version(input_data_scalar)
    assert np.allclose(torch_result_scalar["result"], tf_result_scalar["result"], atol=A_TOL), "Results do not match (scalar)"

    print("Success")

if __name__ == "__main__":
    main()