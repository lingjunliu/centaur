import numpy as np
import torch
import tensorflow as tf

def torch_version(input_dict, cpu=True):
    size = input_dict["size"]
    generator = input_dict.get("generator", None)
    dtype = input_dict.get("dtype", None)
    layout = input_dict.get("layout", torch.strided)
    requires_grad = input_dict.get("requires_grad", False)
    pin_memory = input_dict.get("pin_memory", False)
    device = input_dict.get("device", None)

    if not cpu:
        if generator is not None:
            generator.manual_seed(0)
        if device is None:
            device = "cuda"

    if device is not None:
        device = torch.device(device)
    
    kwargs = {}
    if generator is not None: kwargs['generator'] = generator
    if dtype is not None: kwargs['dtype'] = dtype
    if layout is not torch.strided: kwargs['layout'] = layout
    if device is not None: kwargs['device'] = device
    if requires_grad: kwargs['requires_grad'] = requires_grad
    if pin_memory: kwargs['pin_memory'] = pin_memory

    result = torch.randn(*size, **kwargs)

    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    size = input_dict["size"]
    dtype = input_dict.get("dtype", None)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        if dtype is not None:
             tf_dtype = tf.as_dtype(dtype)
             result = tf.random.normal(size, mean=0.0, stddev=1.0, dtype=tf_dtype)
        else:
            result = tf.random.normal(size, mean=0.0, stddev=1.0)
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    # Example input
    input_data = {
        "size": (2, 3),
        "dtype": np.float32
    }

    # Torch example
    torch_result = torch_version(input_data)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    
    # Assert to see if they are equal
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()