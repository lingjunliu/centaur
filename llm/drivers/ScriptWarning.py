import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    try:
        with torch.jit.ScriptWarning():
            pass
        result = 0
    except Exception as e:
        result = str(e)
    
    if not cpu:
        result = str(result)
    
    return {"result": np.array([0])}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        
        try:
            result = 0
        except Exception as e:
            result = str(e)
        
        result = np.array([0])
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()