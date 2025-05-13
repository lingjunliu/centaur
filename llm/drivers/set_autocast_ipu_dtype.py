import numpy as np
import torch

def torch_version(input_dict, cpu=True):

    input_tensor = torch.tensor(input_dict["input"])
    ipu_dtype_str = input_dict.get("ipu_dtype", "float16")

    if ipu_dtype_str == "float16":
        ipu_dtype = torch.float16
    elif ipu_dtype_str == "float32":
        ipu_dtype = torch.float32
    elif ipu_dtype_str == "bfloat16":
        ipu_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported ipu_dtype: {ipu_dtype_str}")


    if not cpu:
        input_tensor = input_tensor.cuda()

    torch.set_autocast_ipu_dtype(ipu_dtype)
    result = input_tensor

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_np = input_dict["input"]
    ipu_dtype = input_dict.get("ipu_dtype", "float16")

    if ipu_dtype == "float16":
        tf_dtype = tf.float16
    elif ipu_dtype == "float32":
        tf_dtype = tf.float32
    elif ipu_dtype == "bfloat16":
        tf_dtype = tf.bfloat16
    else:
        raise ValueError(f"Unsupported ipu_dtype: {ipu_dtype}")
    
    input_tensor = tf.constant(input_np, dtype=tf_dtype)

    result = input_tensor.numpy()
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "ipu_dtype": "float16"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "ipu_dtype": "float32"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()