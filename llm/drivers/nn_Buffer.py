import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    persistent = input_dict.get("persistent", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    buffer = torch.nn.parameter.UninitializedBuffer(input_tensor.size(), dtype=input_tensor.dtype)
    buffer.materialize(input_tensor)
    
    if not cpu:
        buffer = buffer.cpu()
    
    return {"result": buffer.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        persistent = input_dict.get("persistent", False)
        
        variable = tf.Variable(tf.zeros_like(input_tensor), trainable=False, use_resource=True, name='uninitialized_buffer')
        variable.assign(input_tensor)
        result = variable.read_value()
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "persistent": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "persistent": False
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()