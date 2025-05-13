import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    low = input_dict.get("low", 0)
    high = input_dict["high"]
    size = input_dict["size"]
    generator = input_dict.get("generator", None)
    dtype = input_dict.get("dtype", None)
    layout = input_dict.get("layout", torch.strided)
    requires_grad = input_dict.get("requires_grad", False)

    if generator is not None:
        generator = torch.Generator()
        generator.manual_seed(input_dict["generator"])

    if not cpu:
        if generator is not None:
            generator = torch.Generator()
            generator.manual_seed(input_dict["generator"])
            
    result = torch.randint(low=low, high=high, size=size, generator=generator, dtype=dtype, layout=layout, requires_grad=requires_grad)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    low = input_dict.get("low", 0)
    high = input_dict["high"]
    size = input_dict["size"]
    dtype = input_dict.get("dtype", tf.int64)
    generator = input_dict.get("generator", None)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        if generator is not None:
            result = tf.random.stateless_uniform(shape=size, seed=[generator, 0], minval=low, maxval=high, dtype=dtype)
        else:
            result = tf.random.uniform(shape=size, minval=low, maxval=high, dtype=dtype)
        
        result = tf.cast(result, dtype=dtype).numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "low": 3,
        "high": 10,
        "size": (2, 2),
        "generator": 42
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "high": 10,
        "size": (2, 2),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "low": 3,
        "high": 5,
        "size": (3,),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "low": 3,
        "high": 10,
        "size": (2, 2),
    }

    torch_result = torch_version(input_data, cpu=False)
    tf_result = tensorflow_version(input_data, cpu=False)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()