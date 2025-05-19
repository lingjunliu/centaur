import numpy as np
import array
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

def torch_version(input_dict, cpu=True):

    buffer = input_dict["buffer"]
    dtype = input_dict["dtype"]
    count = input_dict.get("count", -1)
    offset = input_dict.get("offset", 0)
    requires_grad = input_dict.get("requires_grad", False)

    if not cpu:
        pass

    result = torch.frombuffer(buffer, dtype=dtype, count=count, offset=offset, requires_grad=requires_grad)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    buffer = input_dict["buffer"]
    dtype = input_dict["dtype"]
    count = input_dict.get("count", -1)
    offset = input_dict.get("offset", 0)
    requires_grad = input_dict.get("requires_grad", False)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        np_buffer = np.frombuffer(buffer, dtype=np.int8)
        if offset > 0:
            np_buffer = np_buffer[offset:]

        if count > 0:
            np_buffer = np_buffer[:count * np.dtype(torch_to_numpy_dtype(dtype)).itemsize]

        np_result = np.frombuffer(np_buffer, dtype=torch_to_numpy_dtype(dtype))

        result = np_result

    return {"result": result}

def torch_to_numpy_dtype(torch_dtype):
    if torch_dtype == torch.int32:
        return np.int32
    elif torch_dtype == torch.int64:
        return np.int64
    elif torch_dtype == torch.float32:
        return np.float32
    elif torch_dtype == torch.float64:
        return np.float64
    elif torch_dtype == torch.int8:
        return np.int8
    elif torch_dtype == torch.uint8:
        return np.uint8
    elif torch_dtype == torch.int16:
        return np.int16
    elif torch_dtype == torch.float16:
        return np.float16
    else:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")

def main():
    A_TOL = 0.01

    a = array.array('i', [1, 2, 3, 4, 5])

    input_data = {
        "buffer": a,
        "dtype": torch.int32,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    a = array.array('b', [-1, 0, 0, 0])
    input_data = {
        "buffer": a,
        "dtype": torch.int32
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()