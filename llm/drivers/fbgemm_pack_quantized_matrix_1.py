import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    qmin = -128
    qmax = 127
    
    scale = torch.max(torch.abs(input_tensor)) / max(abs(qmin), abs(qmax))
    zero_point = 0

    quantized_tensor = torch.round(input_tensor / scale + zero_point).char()
    
    if not cpu:
        quantized_tensor = quantized_tensor.cpu()

    return {"result": quantized_tensor.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    
    input_tensor = input_dict["input"]
    
    if not isinstance(input_tensor, np.ndarray):
        input_tensor = np.array(input_tensor)

    qmin = np.iinfo(np.int8).min
    qmax = np.iinfo(np.int8).max

    scale = np.max(np.abs(input_tensor)) / (max(abs(qmin), abs(qmax)))
    zero_point = 0

    quantized_tensor = np.round(input_tensor / scale + zero_point).astype(np.int8)
    
    return {"result": quantized_tensor}

def main():
    A_TOL = 1.0

    input_data = {
        "input": np.array([[0.0202, 1.0985], [1.3506, -0.6056]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"].astype(np.float32), tf_result["result"].astype(np.float32), atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()