import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

def torch_version(input_dict, cpu=True):
    dtype_map = {
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.int64: torch.int64,
        np.int32: torch.int32,
        np.uint8: torch.uint8,
        np.int8: torch.int8,
        np.bool_: torch.bool
    }
    
    data_ptr_np = input_dict["data_ptr"]
    size = input_dict.get("size", len(data_ptr_np))
    dtype = dtype_map[input_dict["dtype"]]
    layout = input_dict.get("layout", torch.strided)
    requires_grad = input_dict.get("requires_grad", False)

    data_tensor = torch.tensor(data_ptr_np, dtype=dtype)
    if not cpu:
        data_tensor = data_tensor.cuda()

    storage = data_tensor.untyped_storage()
    
    result = data_tensor.tolist()
    
    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    dtype_map = {
        np.float32: tf.float32,
        np.float64: tf.float64,
        np.int64: tf.int64,
        np.int32: tf.int32,
        np.uint8: tf.uint8,
        np.int8: tf.int8,
        np.bool_: tf.bool
    }
    
    data_ptr_np = input_dict["data_ptr"]
    size = input_dict.get("size", len(data_ptr_np))
    dtype = dtype_map[input_dict["dtype"]]
    layout = input_dict.get("layout", "strided")
    requires_grad = input_dict.get("requires_grad", False)
    
    data_tensor = tf.constant(data_ptr_np, dtype=dtype)

    if not cpu:
        with tf.device('/GPU:0'):
            data_tensor = tf.identity(data_tensor)

    result = data_tensor.numpy().tolist()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    
    data_ptr_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    
    input_data = {
        "data_ptr": data_ptr_np,
        "size": 3,
        "dtype": np.float32,
        "requires_grad": False
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(np.array(torch_result["result"]), np.array(tf_result["result"]), atol=A_TOL), "Results do not match"

    input_data = {
        "data_ptr": data_ptr_np,
        "size": 3,
        "dtype": np.float32,
        "requires_grad": False
    }

    torch_result = torch_version(input_data, cpu=False)
    tf_result = tensorflow_version(input_data, cpu=False)
    
    assert np.allclose(np.array(torch_result["result"]), np.array(tf_result["result"]), atol=A_TOL), "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()