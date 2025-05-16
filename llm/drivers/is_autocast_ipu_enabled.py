import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    if not hasattr(torch, 'is_autocast_ipu_enabled'):
        return {"result": False}
    
    if not cpu:
        torch.cuda.init()

    result = torch.is_autocast_ipu_enabled()
    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    return {"result": np.array(False)}

def main():
    A_TOL = 0.01
    input_data = {}

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()