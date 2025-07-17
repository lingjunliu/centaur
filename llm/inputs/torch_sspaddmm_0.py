
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def sspaddmm_inputs():
    list_of_inputs = []

    # Input 1
    indices_np = np.array([[0, 1], [1, 0]])
    values_np = np.array([1.0, 2.0], dtype=np.float32)
    input_np = torch.sparse_coo_tensor(indices=torch.tensor(indices_np).t(),
                                       values=torch.tensor(values_np),
                                       size=(2, 2))
    mat1_np = torch.sparse_coo_tensor(indices=torch.tensor(indices_np).t(),
                                      values=torch.tensor([3.0, 4.0], dtype=np.float32),
                                      size=(2, 2))
    mat2_np = torch.tensor([[5.0, 6.0], [7.0, 8.0]]).numpy()
    beta_val = 0.5
    alpha_val = 0.25
    out_np = torch.tensor([]).numpy()

    input_dict = {
        "input": input_np,
        "mat1": mat1_np,
        "mat2": mat2_np,
        "beta": beta_val,
        "alpha": alpha_val,
        "out": out_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    indices_np = np.array([[0, 0], [1, 1]])
    values_np = np.array([1.0, 1.0], dtype=np.float32)
    input_np = torch.sparse_coo_tensor(indices=torch.tensor(indices_np).t(),
                                       values=torch.tensor(values_np),
                                       size=(2, 2))
    mat1_np = torch.sparse_coo_tensor(indices=torch.tensor(indices_np).t(),
                                      values=torch.tensor([2.0, 2.0], dtype=np.float32),
                                      size=(2, 2))
    mat2_np = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).numpy()
    beta_val = 1.0
    alpha_val = 1.0
    out_np = torch.tensor([]).numpy()
    input_dict = {
        "input": input_np,
        "mat1": mat1_np,
        "mat2": mat2_np,
        "beta": beta_val,
        "alpha": alpha_val,
        "out": out_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    indices_np = np.array([[0, 0], [0, 1]])
    values_np = np.array([0.5, 0.5], dtype=np.float32)
    input_np = torch.sparse_coo_tensor(indices=torch.tensor(indices_np).t(),
                                       values=torch.tensor(values_np),
                                       size=(1, 2))
    mat1_np = torch.sparse_coo_tensor(indices=torch.tensor(indices_np).t(),
                                      values=torch.tensor([1.0, 1.0], dtype=np.float32),
                                      size=(1, 2))
    mat2_np = torch.tensor([[2.0], [3.0]]).numpy()
    beta_val = 0.0
    alpha_val = 2.0
    out_np = torch.tensor([]).numpy()
    input_dict = {
        "input": input_np,
        "mat1": mat1_np,
        "mat2": mat2_np,
        "beta": beta_val,
        "alpha": alpha_val,
        "out": out_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    indices_np = np.array([[0, 0]])
    values_np = np.array([1.0], dtype=np.float32)
    input_np = torch.sparse_coo_tensor(indices=torch.tensor(indices_np).t(),
                                       values=torch.tensor(values_np),
                                       size=(1, 1))
    mat1_np = torch.sparse_coo_tensor(indices=torch.tensor(indices_np).t(),
                                      values=torch.tensor([1.0], dtype=np.float32),
                                      size=(1, 1))
    mat2_np = torch.tensor([[1.0]]).numpy()
    beta_val = -1.0
    alpha_val = -1.0
    out_np = torch.tensor([]).numpy()
    input_dict = {
        "input": input_np,
        "mat1": mat1_np,
        "mat2": mat2_np,
        "beta": beta_val,
        "alpha": alpha_val,
        "out": out_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    indices_np = np.array([[0, 0], [1, 1]])
    values_np = np.array([1.0, 1.0], dtype=np.float32)
    input_np = torch.sparse_coo_tensor(indices=torch.tensor(indices_np).t(),
                                       values=torch.tensor(values_np),
                                       size=(2, 2))
    mat1_np = torch.sparse_coo_tensor(indices=torch.tensor(indices_np).t(),
                                      values=torch.tensor([2.0, 2.0], dtype=np.float32),
                                      size=(2, 2))
    mat2_np = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).numpy()
    beta_val = 0.2
    alpha_val = 0.8
    out_np = torch.tensor([]).numpy()
    input_dict = {
        "input": input_np,
        "mat1": mat1_np,
        "mat2": mat2_np,
        "beta": beta_val,
        "alpha": alpha_val,
        "out": out_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    indices_np = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    values_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_np = torch.sparse_coo_tensor(indices=torch.tensor(indices_np).t(),
                                       values=torch.tensor(values_np),
                                       size=(2, 2))
    mat1_np = torch.sparse_coo_tensor(indices=torch.tensor(indices_np).t(),
                                      values=torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
                                      size=(2, 2))
    mat2_np = torch.tensor([[9.0, 10.0], [11.0, 12.0]]).numpy()
    beta_val = 0.7
    alpha_val = 0.3
    out_np = torch.tensor([]).numpy()
    input_dict = {
        "input": input_np,
        "mat1": mat1_np,
        "mat2": mat2_np,
        "beta": beta_val,
        "alpha": alpha_val,
        "out": out_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    indices_np = np.array([[0, 0]])
    values_np = np.array([1.0], dtype=np.float32)
    input_np = torch.sparse_coo_tensor(indices=torch.tensor(indices_np).t(),
                                       values=torch.tensor(values_np),
                                       size=(1, 1))
    mat1_np = torch.sparse_coo_tensor(indices=torch.tensor(indices_np).t(),
                                      values=torch.tensor([2.0], dtype=np.float32),
                                      size=(1, 1))
    mat2_np = torch.tensor([[3.0]]).numpy()
    beta_val = -0.5
    alpha_val = 1.5
    out_np = torch.tensor([]).numpy()
    input_dict = {
        "input": input_np,
        "mat1": mat1_np,
        "mat2": mat2_np,
        "beta": beta_val,
        "alpha": alpha_val,
        "out": out_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    indices_np = np.array([[0, 0], [1, 0], [1, 1]])
    values_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_np = torch.sparse_coo_tensor(indices=torch.tensor(indices_np).t(),
                                       values=torch.tensor(values_np),
                                       size=(2, 2))
    mat1_np = torch.sparse_coo_tensor(indices=torch.tensor(indices_np).t(),
                                      values=torch.tensor([4.0, 5.0, 6.0], dtype=np.float32),
                                      size=(2, 2))
    mat2_np = torch.tensor([[7.0, 8.0], [9.0, 10.0]]).numpy()
    beta_val = 2.0
    alpha_val = 0.1
    out_np = torch.tensor([]).numpy()
    input_dict = {
        "input": input_np,
        "mat1": mat1_np,
        "mat2": mat2_np,
        "beta": beta_val,
        "alpha": alpha_val,
        "out": out_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9
    indices_np = np.array([[0, 0], [0, 1]])
    values_np = np.array([1.0, 1.0], dtype=np.float32)
    input_np = torch.sparse_coo_tensor(indices=torch.tensor(indices_np).t(),
                                       values=torch.tensor(values_np),
                                       size=(1, 2))
    mat1_np = torch.sparse_coo_tensor(indices=torch.tensor(indices_np).t(),
                                      values=torch.tensor([2.0, 2.0], dtype=np.float32),
                                      size=(1, 2))
    mat2_np = torch.tensor([[1.0], [3.0]]).numpy()
    beta_val = -0.3
    alpha_val = 0.7
    out_np = torch.tensor([]).numpy()
    input_dict = {
        "input": input_np,
        "mat1": mat1_np,
        "mat2": mat2_np,
        "beta": beta_val,
        "alpha": alpha_val,
        "out": out_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    indices_np = np.array([[0, 0]])
    values_np = np.array([1.0], dtype=np.float32)
    input_np = torch.sparse_coo_tensor(indices=torch.tensor(indices_np).t(),
                                       values=torch.tensor(values_np),
                                       size=(1, 1))
    mat1_np = torch.sparse_coo_tensor(indices=torch.tensor(indices_np).t(),
                                      values=torch.tensor([2.0], dtype=np.float32),
                                      size=(1, 1))
    mat2_np = torch.tensor([[3.0]]).numpy()
    beta_val = 0.9
    alpha_val = -0.4
    out_np = torch.tensor([]).numpy()
    input_dict = {
        "input": input_np,
        "mat1": mat1_np,
        "mat2": mat2_np,
        "beta": beta_val,
        "alpha": alpha_val,
        "out": out_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    indices_np = np.array([[0, 0], [1, 0]])
    values_np = np.array([1.0, 2.0], dtype=np.float32)
    input_np = torch.sparse_coo_tensor(indices=torch.tensor(indices_np).t(),
                                        values=torch.tensor(values_np),
                                        size=(2, 1))
    mat1_np = torch.sparse_coo_tensor(indices=torch.tensor(indices_np).t(),
                                       values=torch.tensor([2.0, 3.0], dtype=np.float32),
                                       size=(2, 1))
    mat2_np = torch.tensor([[4.0]]).numpy()
    beta_val = 0.6
    alpha_val = 0.4
    out_np = torch.tensor([]).numpy()
    input_dict = {
        "input": input_np,
        "mat1": mat1_np,
        "mat2": mat2_np,
        "beta": beta_val,
        "alpha": alpha_val,
        "out": out_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.sspaddmm"] = sspaddmm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sspaddmm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sspaddmm'.")

check_valid('torch.sspaddmm', generated_inputs['torch.sspaddmm'], lib="torch", suffix=0)
