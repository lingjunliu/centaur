import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    low = input_dict.get("low", 0)
    high = input_dict.get("high")
    generator = input_dict.get("generator", None)
    dtype = input_dict.get("dtype", None)

    if not cpu:
        input_tensor = input_tensor.cuda()

    if high is not None:
        if dtype is not None:
            result = torch.randint_like(input_tensor, low, high, dtype=dtype, generator=generator)
        else:
            result = torch.randint_like(input_tensor, low, high, generator=generator)
    else:
        if dtype is not None:
            result = torch.randint_like(input_tensor, high=low, dtype=dtype, generator=generator)
        else:
            result = torch.randint_like(input_tensor, high=low, generator=generator)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        low = input_dict.get("low", 0)
        high = input_dict.get("high")
        dtype = input_tensor.dtype

        shape = tf.shape(input_tensor)

        if high is not None:
            result = tf.random.uniform(shape, minval=low, maxval=high, dtype=tf.int32, seed=123)
            result = tf.cast(result, dtype)
        else:
            result = tf.random.uniform(shape, minval=0, maxval=low, dtype=tf.int32, seed=123)
            result = tf.cast(result, dtype)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.zeros((2, 3), dtype=np.int32),
        "low": 0,
        "high": 5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.zeros((2, 3), dtype=np.int32),
        "low": 2
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.zeros((2, 3), dtype=np.float32),
        "low": 2
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()