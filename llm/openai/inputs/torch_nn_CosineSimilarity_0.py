
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def cosine_similarity_inputs():
    list_of_inputs = []
    
    # 1: 2D, standard
    input1 = torch.randn(100, 128, dtype=torch.float32).numpy()
    input2 = torch.randn(100, 128, dtype=torch.float32).numpy()
    input_dict = {"dim": 1, "eps": 1e-6, "input1": input1, "input2": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 2: 1D vectors, double precision
    input1 = torch.randn(128, dtype=torch.float64).numpy()
    input2 = torch.randn(128, dtype=torch.float64).numpy()
    input_dict = {"dim": 0, "eps": 1e-8, "input1": input1, "input2": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 3: 3D tensors, compute along dim=1
    input1 = torch.randn(4, 64, 3, dtype=torch.float32).numpy()
    input2 = torch.randn(4, 64, 3, dtype=torch.float32).numpy()
    input_dict = {"dim": 1, "eps": 1e-9, "input1": input1, "input2": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 4: Broadcasting on other dims
    input1 = torch.randn(3, 4, 5, dtype=torch.float32).numpy()
    input2 = torch.randn(1, 4, 1, dtype=torch.float32).numpy()
    input_dict = {"dim": 1, "eps": 1e-8, "input1": input1, "input2": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 5: Negative dim, half precision
    input1 = torch.randn(2, 3, 4, dtype=torch.float16).numpy()
    input2 = torch.randn(2, 3, 4, dtype=torch.float16).numpy()
    input_dict = {"dim": -1, "eps": 1e-7, "input1": input1, "input2": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 6: Broadcasting across leading dims
    input1 = torch.randn(5, 7, dtype=torch.float32).numpy()
    input2 = torch.randn(1, 7, dtype=torch.float32).numpy()
    input_dict = {"dim": 1, "eps": 1e-4, "input1": input1, "input2": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 7: 4D tensors, compute along dim=2
    input1 = torch.randn(3, 4, 5, 6, dtype=torch.float32).numpy()
    input2 = torch.randn(3, 4, 5, 6, dtype=torch.float32).numpy()
    input_dict = {"dim": 2, "eps": 1e-8, "input1": input1, "input2": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 8: Negative dim with broadcasting
    input1 = torch.randn(6, 1, 10, 4, dtype=torch.float32).numpy()
    input2 = torch.randn(1, 3, 10, 1, dtype=torch.float32).numpy()
    input_dict = {"dim": -2, "eps": 1e-5, "input1": input1, "input2": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 9: Zero eps
    input1 = torch.randn(2, 5, 6, dtype=torch.float32).numpy()
    input2 = torch.randn(2, 5, 6, dtype=torch.float32).numpy()
    input_dict = {"dim": 1, "eps": 0.0, "input1": input1, "input2": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 10: Different shapes with broadcasting on non-dim axes
    input1 = torch.randn(1, 3, 1, dtype=torch.float32).numpy()
    input2 = torch.randn(4, 3, 2, dtype=torch.float32).numpy()
    input_dict = {"dim": 1, "eps": 1e-3, "input1": input1, "input2": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 11: dim=0, broadcasting on last dim
    input1 = torch.randn(10, 5, dtype=torch.float32).numpy()
    input2 = torch.randn(10, 1, dtype=torch.float32).numpy()
    input_dict = {"dim": 0, "eps": 1e-8, "input1": input1, "input2": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 12: 5D tensors with broadcasting, compute along dim=1
    input1 = torch.randn(2, 3, 4, 5, 6, dtype=torch.float32).numpy()
    input2 = torch.randn(1, 3, 1, 5, 1, dtype=torch.float32).numpy()
    input_dict = {"dim": 1, "eps": 1e-6, "input1": input1, "input2": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # 13: Negative dim (-3) on 4D with broadcasting
    input1 = torch.randn(3, 4, 5, 6, dtype=torch.float32).numpy()
    input2 = torch.randn(1, 4, 1, 1, dtype=torch.float32).numpy()
    input_dict = {"dim": -3, "eps": 1e-8, "input1": input1, "input2": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.CosineSimilarity"] = cosine_similarity_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.CosineSimilarity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.CosineSimilarity'.")


check_valid('torch.nn.CosineSimilarity', generated_inputs['torch.nn.CosineSimilarity'], lib="torch", suffix=0)
