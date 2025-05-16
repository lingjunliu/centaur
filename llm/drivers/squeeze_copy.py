import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    if "dim" in input_dict:
      dim = input_dict["dim"]
    else:
      dim = None
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    if dim is not None:
        result = torch.squeeze(input_tensor, dim=dim)
    else:
        result = torch.squeeze(input_tensor)
    
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
        if "dim" in input_dict:
            dim = input_dict["dim"]
            try:
                result = tf.squeeze(input_tensor, axis=dim)
            except:
                result = input_tensor
        else:
            result = tf.squeeze(input_tensor)
            
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data1 = {
        "input": np.array([[[1, 2, 3]]], dtype=np.float32),
    }
    
    torch_result1 = torch_version(input_data1)
    tf_result1 = tensorflow_version(input_data1)
    
    assert np.allclose(torch_result1["result"], tf_result1["result"], atol=A_TOL), "Results do not match"
    
    input_data2 = {
        "input": np.array([[[1, 2, 3]]], dtype=np.float32),
        "dim": 0
    }
    
    torch_result2 = torch_version(input_data2)
    tf_result2 = tensorflow_version(input_data2)
    
    assert np.allclose(torch_result2["result"], tf_result2["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()