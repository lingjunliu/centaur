import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    min = input_dict.get("min", None)
    max = input_dict.get("max", None)

    if not cpu:
        input_tensor = input_tensor.cuda()
        if min is not None:
            min = torch.tensor(min)
            if not cpu:
                min = min.cuda()
        if max is not None:
            max = torch.tensor(max)
            if not cpu:
                max = max.cuda()
    
    result = torch.clamp(input_tensor, min=min, max=max)
    
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
        min = input_dict.get("min", None)
        max = input_dict.get("max", None)

        if min is not None:
           min = tf.constant(min, dtype=input_tensor.dtype)
        if max is not None:
           max = tf.constant(max, dtype=input_tensor.dtype)
           
        result = tf.clip_by_value(input_tensor, clip_value_min=min if min is not None else tf.float32.min, clip_value_max=max if max is not None else tf.float32.max)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        "min": 0.0,
        "max": 1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        "min": -0.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        "max": 1.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    

    print("Success")

if __name__ == "__main__":
    main()