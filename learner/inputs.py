import torch
import copy

def scatter_inputs():
    list_of_inputs = []
    # Input 1, valid
    src_torch = torch.arange(1, 11).reshape((2, 5))
    src = src_torch.numpy() 
    index = torch.tensor([[0, 1, 2, 0]]).numpy()
    input = torch.zeros(3, 5, dtype=src_torch.dtype).numpy()
    dim = 0

    input_dict = {
        "input": input,
        "dim": dim,
        "src": src, 
        "index": index,         
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, invalid
    index_2 = torch.tensor([[0, 1, 2], [0, 1, 4]])
    input_dict["index"] = index_2.numpy()
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    input_dict["dim"] = 1
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, invalid
    input_dict = {
        "input": torch.full((2, 4), 2.).numpy(),
        "dim": 1,
        "src": 1.23, 
        "index": torch.tensor([[2], [3]]).numpy(),         
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    input_dict = {
        "input": torch.full((2, 4), 2., dtype=int).numpy(),
        "dim": 1,
        "src": torch.tensor([[2], [3]]).numpy(), 
        "index": torch.tensor([[2], [3]]).numpy(),         
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input_dict = {
        "input": torch.full((2, 4), 2., dtype=torch.float32).numpy(),
        "dim": 1,
        "src": torch.tensor([[2], [3]], dtype=torch.float32).numpy(), 
        "index": torch.tensor([[2], [3]]).numpy(),         
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

def conv_transpose2d_inputs():
    list_of_inputs = []
    # Input 1, valid
    input = torch.randn(1, 3, 5, 5).numpy()        # [N, C_in, H_in, W_in]
    weight = torch.randn(3, 2, 3, 3).numpy()       # [C_in, C_out, kH, kW]
    stride = 1
    padding = 0
    
    input_dict = {
        "input": input,
        "weight": weight,
        "stride": stride,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    input = torch.randn(1, 4, 4, 4).numpy()        # [N, C_in, H_in, W_in]
    weight = torch.randn(4, 2, 3, 3).numpy()       # [C_in, C_out, kH, kW]
    stride = 2
    padding = 1
    
    input_dict = {
        "input": input,
        "weight": weight,
        "stride": stride,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    input = torch.randn(1, 2, 6, 6).numpy()        # [N, C_in, H_in, W_in]
    weight = torch.randn(2, 1, 5, 5).numpy()       # [C_in, C_out, kH, kW]
    stride = 1
    padding = 0
    
    input_dict = {
        "input": input,
        "weight": weight,
        "stride": stride,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    input = torch.randn(1, 1, 5, 5).numpy()        # [N, C_in, H_in, W_in]
    weight = torch.randn(1, 1, 3, 3).numpy()       # [C_in, C_out, kH, kW]
    stride = 2
    padding = 1
    
    input_dict = {
        "input": input,
        "weight": weight,
        "stride": stride,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    input = torch.randn(2, 8, 10, 10).numpy()        # [N, C_in, H_in, W_in]
    weight = torch.randn(8, 16, 4, 4).numpy()       # [C_in, C_out, kH, kW]
    stride = 2
    padding = 1
    
    input_dict = {
        "input": input,
        "weight": weight,
        "stride": stride,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

inputs_per_api = {
    "scatter": scatter_inputs,
    "conv_transpose2d": conv_transpose2d_inputs
}

def get_inputs(api):
    if api not in inputs_per_api:
        raise NotImplementedError(f"No inputs added for {api}")
    return inputs_per_api[api]()