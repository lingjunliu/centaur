import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    dtype = input_dict["dtype"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.can_cast(input_tensor.dtype, dtype)

    if not cpu:
        pass

    return {"result": np.array(result)}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    import torch

    input_torch_dtype = torch.tensor(input_dict["input"]).dtype
    target_torch_dtype = input_dict["dtype"]

    input_tf_dtype = tf.as_dtype(str(input_torch_dtype).split('.')[-1])
    target_tf_dtype = tf.as_dtype(str(target_torch_dtype).split('.')[-1])

    try:
        tf.cast(tf.zeros((), dtype=input_tf_dtype), dtype=target_tf_dtype)
        can_cast = True
    except (tf.errors.InvalidArgumentError, ValueError, TypeError):
        can_cast = False
    except Exception as e:
        can_cast = False

    return {"result": np.array(can_cast)}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3], dtype=np.int32),
        "dtype": torch.float32
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3], dtype=np.float32),
        "dtype": torch.int32
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3], dtype=np.float32),
        "dtype": torch.uint8
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3], dtype=np.int8),
        "dtype": torch.float64
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3], dtype=np.float16),
        "dtype": torch.float64
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3], dtype=np.float64),
        "dtype": torch.complex64
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"



    print("Success")


if __name__ == "__main__":
    import torch
    main()