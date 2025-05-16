import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    generator = input_dict.get("generator", None)
    if generator is not None:
        generator = torch.Generator().manual_seed(int(generator))
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    if generator is not None:
        result = torch.rand_like(input_tensor, generator=generator)
    else:
        result = torch.rand_like(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        seed = input_dict.get("generator", None)
        
        if seed is not None:
            generator = tf.random.Generator.from_seed(seed)
            result = generator.uniform(shape=input_tensor.shape, dtype=input_tensor.dtype)
        else:
            result = tf.random.uniform(shape=input_tensor.shape, dtype=input_tensor.dtype)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "generator": 42
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    gen = np.random.default_rng(42)
    expected = gen.random(size=input_data['input'].shape, dtype=input_data['input'].dtype)
    assert np.allclose(tf_result['result'], expected, atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data, cpu=False)
    tf_result = tensorflow_version(input_data, cpu=False)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()