import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    size = input_dict["size"]
    dtype = input_dict.get("dtype", None)
    layout = input_dict.get("layout", torch.strided)
    requires_grad = input_dict.get("requires_grad", False)
    pin_memory = input_dict.get("pin_memory", False)
    memory_format = input_dict.get("memory_format", torch.contiguous_format)
    
    if dtype is not None:
      dtype = getattr(torch, dtype)

    if not cpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    result = torch.empty(*size, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, pin_memory=pin_memory, memory_format=memory_format)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    size = input_dict["size"]
    dtype = input_dict.get("dtype", None)
    requires_grad = input_dict.get("requires_grad", False)

    if dtype is not None:
      dtype = getattr(tf, dtype)
    else:
      dtype = tf.float32

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        result = tf.Variable(tf.zeros(size, dtype=dtype), trainable=requires_grad)
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "size": (2, 3),
        "dtype": "int64",
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "size": (2, 3),
        "dtype": "float32",
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()