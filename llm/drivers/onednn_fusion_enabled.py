import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    flag = input_dict["flag"]

    if flag:
        torch.jit.onednn_fusion_enabled()
    else:
        torch.jit.onednn_fusion_enabled()


    return {"result": torch.jit.is_onednn_fusion_enabled()}

def tensorflow_version(input_dict, cpu=True):
    return {"result": None}

def main():
    A_TOL = 0.01
    input_data = {
        "flag": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    assert torch_result["result"] == torch.jit.is_onednn_fusion_enabled()

    print("Success")

if __name__ == "__main__":
    main()