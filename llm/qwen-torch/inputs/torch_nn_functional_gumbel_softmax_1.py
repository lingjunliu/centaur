
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def gumbel_softmax_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    logits = torch.tensor([0.1, 0.2, 0.3]).numpy()  # tensor
    tau = 0.5  # float
    hard = True  # boolean
    dim = 0  # integer
    
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    logits = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]]).numpy()  # tensor
    tau = 1.0  # float
    hard = False  # boolean
    dim = 1  # integer
    
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    logits = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()  # tensor
    tau = 0.1  # float
    hard = True  # boolean
    dim = 2  # integer
    
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    logits = torch.tensor([-0.1, -0.2, -0.3]).numpy()  # tensor
    tau = 0.5  # float
    hard = False  # boolean
    dim = 0  # integer
    
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()  # tensor
    tau = 0.7  # float
    hard = True  # boolean
    dim = 0  # integer
    
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    logits = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()  # tensor
    tau = 0.3  # float
    hard = False  # boolean
    dim = 1  # integer
    
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    logits = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]).numpy()  # tensor
    tau = 0.8  # float
    hard = True  # boolean
    dim = 2  # integer
    
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    logits = torch.tensor([-1.0, -2.0, -3.0]).numpy()  # tensor
    tau = 0.2  # float
    hard = True  # boolean
    dim = 0  # integer
    
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    logits = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()  # tensor
    tau = 0.9  # float
    hard = False  # boolean
    dim = 1  # integer
    
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    logits = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]).numpy()  # tensor
    tau = 0.4  # float
    hard = True  # boolean
    dim = 2  # integer
    
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.gumbel_softmax_1"] = gumbel_softmax_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.gumbel_softmax_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.gumbel_softmax_1'.")


check_valid('torch.nn.functional.gumbel_softmax', generated_inputs['torch.nn.functional.gumbel_softmax_1'], lib="torch", suffix=1)
