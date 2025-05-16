import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    qtensor = torch.tensor(input_dict["qtensor"])
    scale = input_dict.get("scale", 1.0)
    zero_point = input_dict.get("zero_point", 0)
    dtype_str = input_dict.get("dtype", "torch.quint8")
    dtype = eval(dtype_str)
    
    if not cpu:
        qtensor = qtensor.cuda()
    
    result = torch.quantize_per_tensor(torch.empty(tuple(qtensor.shape)), scale=scale, zero_point=zero_point, dtype=dtype)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    qtensor_np = input_dict["qtensor"]
    scale = input_dict.get("scale", 1.0)
    zero_point = input_dict.get("zero_point", 0)
    dtype = input_dict.get("dtype", tf.quint8) 

    qtensor = tf.constant(qtensor_np)

    shape = qtensor.shape.as_list()

    if dtype == tf.quint8:
        result = tf.zeros(shape, dtype=tf.uint8).numpy()
    elif dtype == tf.qint8:
        result = tf.zeros(shape, dtype=tf.int8).numpy()
    elif dtype == tf.qint32:
        result = tf.zeros(shape, dtype=tf.int32).numpy()
    else:
        raise ValueError("Unsupported dtype")

    return {"result": result}

def main():
    A_TOL = 0.01
    
    input_data = {
        "qtensor": np.array([1, 2, 3, 4], dtype=np.int8),
        "scale": 2.0,
        "zero_point": 1,
        "dtype": "torch.quint8"
    }

    torch_result = torch_version(input_data)
    
    tf_input_data = {
        "qtensor": np.array([1, 2, 3, 4], dtype=np.int8),
        "scale": 2.0,
        "zero_point": 1
    }
    tf_result = tensorflow_version(tf_input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()