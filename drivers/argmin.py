def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    
    # Unpack input dictionary
    input_tensor = torch.tensor(input_dict["input"])
    dim = input_dict.get("dim", None)

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply to torch.argmin
    result = torch.argmin(input_tensor, dim=dim)

    if not cpu:
        result = result.cpu()

    return {"argmin_result": result.numpy()}