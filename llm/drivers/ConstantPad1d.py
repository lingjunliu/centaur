import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict["padding"]
    value = input_dict.get("value", 0.0)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    pad1d = torch.nn.ConstantPad1d(padding, value)
    result = pad1d(input_tensor)
    
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
        padding = input_dict["padding"]
        value = input_dict.get("value", 0.0)

        input_shape = tf.shape(input_tensor)
        input_rank = tf.rank(input_tensor)
        
        if isinstance(padding, int):
            pad_left = padding
            pad_right = padding
        else:
            pad_left = padding[0]
            pad_right = padding[1]
            
        paddings = [[0, 0] for _ in range(input_rank - 1)]
        paddings.append([pad_left, pad_right])
        paddings = tf.constant(paddings, dtype=tf.int32)
            
        constant_values = tf.constant(value, dtype=input_tensor.dtype)
        result = tf.pad(input_tensor, paddings, "CONSTANT", constant_values=constant_values)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3], dtype=np.float32),
        "padding": (2, 3),
        "value": 1.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3], dtype=np.float32),
        "padding": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()