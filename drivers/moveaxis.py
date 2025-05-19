import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    source = input_dict["source"]
    destination = input_dict["destination"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.moveaxis(input_tensor, source, destination)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    source = input_dict["source"]
    destination = input_dict["destination"]

    input_tensor_np = input_tensor.numpy()
    source = source if isinstance(source, tuple) else (source,)
    destination = destination if isinstance(destination, tuple) else (destination,)

    axes = list(range(len(input_tensor_np.shape)))

    for s, d in zip(source, destination):
        axes.insert(d, axes.pop(axes.index(s)))
    
    result = tf.transpose(input_tensor, perm=axes)
    result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(3, 2, 1).astype(np.float32),
        "source": 1,
        "destination": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(3, 2, 1).astype(np.float32),
        "source": (1, 2),
        "destination": (0, 1)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()