import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.is_conj(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": np.array([result], dtype=np.float32)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        
        result = tf.math.equal(tf.math.imag(input_tensor), 0.0)

        result = tf.reduce_all(result).numpy()
        
    return {"result": np.array([result], dtype=np.float32)}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1+0j, 2+0j, 3+0j, 4+0j], dtype=np.complex64)
    }

    torch_result = torch_version(input_data)
    
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1+1j, 2+0j, 3+0j, 4+0j], dtype=np.complex64)
    }

    torch_result = torch_version(input_data)
    
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()