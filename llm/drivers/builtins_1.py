import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    start = input_dict["start"]
    length = input_dict.get("length", None)
    fill_value = input_dict.get("fill_value", 0)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.masked_fill(input_tensor, input_tensor.abs() > start, fill_value)
    
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
        start = input_dict["start"]
        length = input_dict.get("length", None)
        fill_value = input_dict.get("fill_value", 0)
        
        mask = tf.math.abs(input_tensor) > start
        result = tf.where(mask, tf.fill(tf.shape(input_tensor), tf.cast(fill_value, input_tensor.dtype)), input_tensor)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "start": 1.0,
        "fill_value": -1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()