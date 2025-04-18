def torch_version(input, cpu=True):
    import torch
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["other"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        other_tensor = other_tensor.cuda()
    
    # Apply torch.matmul
    result = torch.matmul(input_tensor, other_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"matmul": result.numpy()}