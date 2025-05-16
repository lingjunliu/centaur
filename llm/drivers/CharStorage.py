import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_ = torch.tensor(input_dict["input"])
    
    if not cpu:
        input_ = input_.cuda()
    
    storage = torch.CharStorage.from_buffer(input_.numpy().tobytes())
    
    if not cpu:
        result = storage
    else:
        result = storage
    
    return {"result": np.array(list(result))}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_np = input_dict["input"]
    
    return {"result": input_np}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([65, 66, 67, 68], dtype=np.int8)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()