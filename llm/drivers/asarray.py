import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    obj = input_dict["obj"]
    dtype = input_dict.get("dtype", None)
    copy = input_dict.get("copy", None)
    device = input_dict.get("device", None)
    requires_grad = input_dict.get("requires_grad", False)
    
    if isinstance(obj, np.ndarray):
        obj = torch.tensor(obj)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        obj = torch.tensor(obj)
    elif isinstance(obj, float) or isinstance(obj, int):
        obj = obj
    elif isinstance(obj, torch.Tensor):
        pass
    else:
        raise TypeError("Unsupported type for obj")

    if not cpu:
        if isinstance(obj, torch.Tensor):
            obj = obj.cuda()
        else:
            obj = torch.tensor(obj).cuda()

    if requires_grad and (not isinstance(obj, torch.Tensor) or not obj.is_floating_point()):
        if isinstance(obj, torch.Tensor):
            obj = obj.float()
        else:
            obj = torch.tensor(obj, dtype=torch.float32)
        if not cpu:
            obj = obj.cuda()
            
    result = torch.asarray(obj, dtype=dtype, copy=copy, requires_grad=requires_grad)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    obj = input_dict["obj"]
    dtype = input_dict.get("dtype", None)
    copy = input_dict.get("copy", None)
    device = input_dict.get("device", None)
    requires_grad = input_dict.get("requires_grad", False)
    
    if isinstance(obj, np.ndarray):
        obj = tf.constant(obj)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        obj = tf.constant(obj)
    elif isinstance(obj, float) or isinstance(obj, int):
        obj = tf.constant(obj)
    elif isinstance(obj, tf.Tensor):
        pass
    else:
        raise TypeError("Unsupported type for obj")
    
    if not cpu:
        with tf.device('/GPU:0'):
            if dtype is not None:
                 obj = tf.cast(obj, dtype=dtype)
            result = tf.identity(obj)
            result = result.numpy()
    else:
        if dtype is not None:
             obj = tf.cast(obj, dtype=dtype)
        result = tf.identity(obj)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "obj": np.array([1, 2, 3], dtype=np.int64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "obj": [1, 2, 3],
        "dtype": None,
        "copy": True,
        "requires_grad": True,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "obj": np.array([1.0, 2.0, 3.0]),
        "dtype": None,
        "copy": True,
        "requires_grad": True,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()