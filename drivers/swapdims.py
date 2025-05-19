import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    dim0 = input_dict["dim0"]
    dim1 = input_dict["dim1"]
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.swapdims(input_tensor, dim0, dim1)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    dim0 = input_dict["dim0"]
    dim1 = input_dict["dim1"]

    dims = list(range(len(input_tensor.shape)))
    dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
    result = tf.transpose(input_tensor, perm=dims)
    
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[0,1],[2,3]],[[4,5],[6,7]]], dtype=np.float32),
        "dim0": 0,
        "dim1": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([[[0,1],[2,3]],[[4,5],[6,7]]], dtype=np.float32),
        "dim0": 0,
        "dim1": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()