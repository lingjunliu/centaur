import torch
import copy
import time
from utils.api_utils import get_signatures, get_driver
from utils.misc import get_dir_in_root, get_tmp_dir, create_subdir, save_to_new_pkl, read_pkl
from generator.input_generators import get_random_input, get_abstract_input, concretize_input
from eval.oracle import oracle_crash
import llm.valid_inputs as valid_inputs
import llm.valid_inputs_old as valid_inputs_old
import numpy as np
import os

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
    
    # Skipping invalid inputs
    # list_of_inputs.append(copy.deepcopy(input_dict))

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
    
    # Skipping invalid inputs
    # list_of_inputs.append(copy.deepcopy(input_dict))

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
    
    # Input 6, invalid
    input = torch.randn(2, 8, 10, 10, dtype=torch.complex64).numpy()        # [N, C_in, H_in, W_in]
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
    
    # Input 7, invalid
    input = torch.randn(2, 8, 10, 10).numpy()        # [N, C_in, H_in, W_in]
    weight = torch.randint(low=0, high=100, size=(8, 16, 4, 4), dtype=torch.int64).numpy()       # [C_in, C_out, kH, kW]
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

def matmul_inputs():
    list_of_inputs = []
    # Input 1, valid
    input = torch.randn(3, 5).numpy()
    other = torch.randn(5, 2).numpy() 

    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.randn(3, 5).numpy()
    other = torch.randn(5).numpy()

    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

def add_inputs():
    list_of_inputs = []
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other = torch.tensor([0.1, 0.2, 0.3]).numpy()
    alpha = 10.0

    input_dict = {
        "input": input,
        "other": other,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([1, 2, 3], dtype=torch.int32).numpy() 
    other = torch.tensor([4, 5, 6], dtype=torch.int32).numpy()
    alpha = 2

    input_dict = {
        "input": input,
        "other": other,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

def combinations_inputs():
    list_of_inputs = []
    # Input 1, valid
    input = torch.tensor([10, 20, 30, 40]).numpy()
    r = 3
    with_replacement = False

    input_dict = {
        "input": input,
        "r": r,
        "with_replacement": with_replacement
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([1.5, 2.5]).numpy()
    r = 2
    with_replacement = True

    input_dict = {
        "input": input,
        "r": r,
        "with_replacement": with_replacement
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

def addcmul_inputs():
    list_of_inputs = []
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    tensor1 = torch.tensor([0.1, 0.2, 0.3]).numpy()
    tensor2 = torch.tensor([10.0, 20.0, 30.0]).numpy()
    value = 2.0  

    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    tensor1 = torch.tensor([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]]).numpy()
    tensor2 = torch.tensor([[0.1, 0.2, 0.3],
                            [0.4, 0.5, 0.6]]).numpy()
    value = 0.5

    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

def lp_pool1d_inputs():
    list_of_inputs = []
    # Input 1, valid
    input = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]]).numpy()
    norm_type = 2.0
    kernel_size = 2
    stride = 2
    ceil_mode = True
    
    input_dict = {
        "input": input,
        "norm_type": norm_type,
        "kernel_size": kernel_size,
        "stride": stride,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    input = torch.tensor([[[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]]]).numpy()
    norm_type = 1.0
    kernel_size = 3
    stride = 2
    ceil_mode = True
    
    input_dict = {
        "input": input,
        "norm_type": norm_type,
        "kernel_size": kernel_size,
        "stride": stride,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

def introduce_floats(input_dict, signature):
    """
    Mutation to prevent learning rule_8 incorrectly
    """
    seed = 42
    mutated_inputs = []
    float_types = [np.float16, np.float32, np.float64]
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(float_types))
    for arg, domain in signature.items():
        if input_dict[arg] is None:
            continue
        
        if domain in ["tensor", "tensor_list"]:
            new_input = copy.deepcopy(input_dict)
            if isinstance(new_input[arg], np.ndarray):
                new_input[arg] = new_input[arg].astype(float_types[index%len(float_types)])
            elif isinstance(new_input[arg], list):
                for i, _ in enumerate(new_input[arg]):
                    new_input[arg][i] = new_input[arg][i].astype(float_types[index%len(float_types)])
            else:
                new_input[arg] = float_types[index%len(float_types)](new_input[arg])
            index += 1
            mutated_inputs.append(new_input)
    
    return mutated_inputs

def introduce_integers(input_dict, signature):
    """
    Mutation to prevent learning rule_13 incorrectly
    """
    seed = 42
    mutated_inputs = []
    int_types = [np.int8, np.int16, np.int32, np.int64, np.uint8]
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(int_types))
    for arg, domain in signature.items():
        if input_dict[arg] is None:
            continue
        
        if domain in ["tensor", "tensor_list"]:
            new_input = copy.deepcopy(input_dict)
            if isinstance(new_input[arg], np.ndarray):
                new_input[arg] = new_input[arg].astype(int_types[index%len(int_types)])
            elif isinstance(new_input[arg], list):
                for i, _ in enumerate(new_input[arg]):
                    new_input[arg][i] = new_input[arg][i].astype(int_types[index%len(int_types)])
            else:
                new_input[arg] = int_types[index%len(int_types)](new_input[arg])
            index += 1
            mutated_inputs.append(new_input)
    
    return mutated_inputs

def introduce_empty_tensors(input_dict, signature):
    """
    Mutation to prevent learning rule_14 incorrectly
    """
    mutated_inputs = []
    for arg, domain in signature.items():
        if domain in ["tensor", "tensor_list"]:
            new_input = copy.deepcopy(input_dict)
            new_input[arg] = np.array([])
            mutated_inputs.append(new_input)
    
    return mutated_inputs

def introduce_zeros(input_dict, signature):
    """
    Mutation to prevent learning rule_21 incorrectly
    """
    mutated_inputs = []
    for arg, domain in signature.items():
        new_input = None
        if domain in ["tensor", "tensor_list"]:
            new_input = copy.deepcopy(input_dict)
            if isinstance(new_input[arg], np.ndarray):
                new_input[arg] = np.zeros(new_input[arg].shape)
            else:
                new_input[arg] = 0.0
            
        elif domain == "integer":
            new_input = copy.deepcopy(input_dict)
            new_input[arg] = 0
        elif domain == "float":
            new_input = copy.deepcopy(input_dict)
            new_input[arg] = 0.0
        elif domain in ["list", "tuple"]:
            new_input = copy.deepcopy(input_dict)
            
            if new_input[arg] is None:
                new_input[arg] = [0]
            else:
                if domain == "tuple":
                    new_input[arg] = list(new_input[arg])
                for i, entry in enumerate(new_input[arg]):
                    new_input[arg][i] = 0
                
            if domain == "tuple":
                new_input[arg] = tuple(new_input[arg])
        
        if new_input:
            mutated_inputs.append(new_input)

    return mutated_inputs

def introduce_opposite_bools(input_dict, signature):
    """
    Mutation to increase diversity
    """
    mutated_inputs = []
    for arg, domain in signature.items():
        if domain == "boolean":
            new_input = copy.deepcopy(input_dict)
            new_input[arg] = not new_input[arg]
            mutated_inputs.append(new_input)
            
    return mutated_inputs

def introduce_negatives(input_dict, signature):
    """
    Mutation to prevent learning rule_17 and rule_18 incorrectly
    """
    mutated_inputs = []
    none_replacements = {
        "integer": -1,
        "float": -1.0,
        "list": [-1],
        "tuple": (-1),
        "tensor": np.array([-1.0]),
        "tensor_list": np.array([-1.0])
    }
    for arg, domain in signature.items():
        new_input = None
        if input_dict[arg] is None and domain in none_replacements:
            new_input = copy.deepcopy(input_dict)
            new_input[arg] = none_replacements[domain]
        elif domain in ["tensor", "tensor_list", "integer", "float"]:
            new_input = copy.deepcopy(input_dict)
            new_input[arg] = new_input[arg] * -1
        elif domain in ["list", "tuple"]:
            new_input = copy.deepcopy(input_dict)
            for i, entry in enumerate(new_input[arg]):
                if domain == "tuple":
                    new_input[arg] = list(new_input[arg])
            
                for i, entry in enumerate(new_input[arg]):
                    if new_input[arg][i] is None:
                        new_input[arg][i] = -1
                    else:
                        new_input[arg][i] = new_input[arg][i] * -1
                    
                if domain == "tuple":
                    new_input[arg] = tuple(new_input[arg])
        
        if new_input:
            mutated_inputs.append(new_input)

    return mutated_inputs

def augment_inputs(list_of_inputs, signature):
    """
    Mutate inputs to have diversity to ensure wrong invariants are not learned
    And return the original inputs + mutated inputs
    """
    mutators = [introduce_empty_tensors, introduce_floats, introduce_integers, introduce_negatives, introduce_opposite_bools, introduce_zeros]
    mutated_inputs = []
    for input_dict in list_of_inputs:
        for arg,domain in signature.items():
            if arg not in input_dict:
                print(f"\nSignature: {signature} | input: {input_dict.keys()}\n")
        for mutator in mutators:
            mutated_inputs += mutator(input_dict, signature)
            
    return list_of_inputs + mutated_inputs

# Add human and LLM defined inputs for APIs that are
# difficult to generate inputs for
inputs_per_api = {
    "scatter": scatter_inputs(),      # human start
    "conv_transpose2d": conv_transpose2d_inputs(),
    "matmul": matmul_inputs(),
    "add": add_inputs(),
    "combinations": combinations_inputs(),
    "addcmul": addcmul_inputs(),
    "lp_pool1d_": lp_pool1d_inputs(), # human end
    'full': valid_inputs.full_inputs() + valid_inputs_old.full_inputs(),    # llm start
    'empty_strided': valid_inputs.empty_strided_inputs() + valid_inputs_old.empty_strided_inputs(),
    'broadcast_shapes': valid_inputs.broadcast_shapes_inputs() + valid_inputs_old.broadcast_shapes_inputs(),
    'rand': valid_inputs.rand_inputs() + valid_inputs_old.rand_inputs(),
    'zeros': valid_inputs.zeros_inputs() + valid_inputs_old.zeros_inputs(),
    'ones': valid_inputs.ones_inputs() + valid_inputs_old.ones_inputs(),
    'max_pool1d': valid_inputs.max_pool1d_inputs() + valid_inputs_old.max_pool1d_inputs(),
    'mean': valid_inputs.mean_inputs() + valid_inputs_old.mean_inputs(),
    'rot90': valid_inputs.rot90_inputs() + valid_inputs_old.rot90_inputs(),
    'max_pool3d': valid_inputs.max_pool3d_inputs() + valid_inputs_old.max_pool3d_inputs(),
    'var': valid_inputs.var_inputs() + valid_inputs_old.var_inputs(),
    'transpose': valid_inputs.transpose_inputs() + valid_inputs_old.transpose_inputs(),
    'max_pool2d': valid_inputs.max_pool2d_inputs() + valid_inputs_old.max_pool2d_inputs(),
    'normalize': valid_inputs.normalize_inputs() + valid_inputs_old.normalize_inputs(),
    'split': valid_inputs.split_inputs() + valid_inputs_old.split_inputs(),
    'var_mean': valid_inputs.var_mean_inputs() + valid_inputs_old.var_mean_inputs(),
    'std': valid_inputs.std_inputs() + valid_inputs_old.std_inputs(),
    'unsqueeze': valid_inputs.unsqueeze_inputs() + valid_inputs_old.unsqueeze_inputs(),
    'nansum': valid_inputs.nansum_inputs() + valid_inputs_old.nansum_inputs(),
    'pixel_shuffle': valid_inputs.pixel_shuffle_inputs() + valid_inputs_old.pixel_shuffle_inputs(),
    'logcumsumexp': valid_inputs.logcumsumexp_inputs() + valid_inputs_old.logcumsumexp_inputs(),
    'movedim': valid_inputs.movedim_inputs() + valid_inputs_old.movedim_inputs(),
    'narrow': valid_inputs.narrow_inputs() + valid_inputs_old.narrow_inputs(),
    'pad': valid_inputs.pad_inputs() + valid_inputs_old.pad_inputs(),
    'reshape': valid_inputs.reshape_inputs() + valid_inputs_old.reshape_inputs(),
    'std_mean': valid_inputs.std_mean_inputs() + valid_inputs_old.std_mean_inputs(),
    'linspace': valid_inputs.linspace_inputs() + valid_inputs_old.linspace_inputs(),
    'logspace': valid_inputs.logspace_inputs() + valid_inputs_old.logspace_inputs(),
    'lstsq': valid_inputs.lstsq_inputs() + valid_inputs_old.lstsq_inputs(),
    'trapz': valid_inputs.trapz_inputs() + valid_inputs_old.trapz_inputs(),
    'triangular_solve': valid_inputs.triangular_solve_inputs() + valid_inputs_old.triangular_solve_inputs(),
    'tensordot': valid_inputs.tensordot_inputs() + valid_inputs_old.tensordot_inputs(),
    'multilabel_soft_margin_loss': valid_inputs.multilabel_soft_margin_loss_inputs() + valid_inputs_old.multilabel_soft_margin_loss_inputs(),
    'linear_': valid_inputs.linear_inputs() + valid_inputs_old.linear_inputs(),
    'lu_solve': valid_inputs.lu_solve_inputs() + valid_inputs_old.lu_solve_inputs(),
    'margin_ranking_loss': valid_inputs.margin_ranking_loss_inputs() + valid_inputs_old.margin_ranking_loss_inputs(),
    'where': valid_inputs.where_inputs() + valid_inputs_old.where_inputs(),
    'pow': valid_inputs.pow_inputs() + valid_inputs_old.pow_inputs(),
    'mm': valid_inputs.mm_inputs() + valid_inputs_old.mm_inputs(),
    'mv': valid_inputs.mv_inputs() + valid_inputs_old.mv_inputs(),
    'polar': valid_inputs.polar_inputs() + valid_inputs_old.polar_inputs(),
    'prelu': valid_inputs.prelu_inputs() + valid_inputs_old.prelu_inputs(),
    'solve': valid_inputs.solve_inputs() + valid_inputs_old.solve_inputs(),
    'sub': valid_inputs.sub_inputs() + valid_inputs_old.sub_inputs(),
    'nll_loss': valid_inputs.nll_loss_inputs() + valid_inputs_old.nll_loss_inputs(),
    'flatten_': valid_inputs.flatten_inputs() + valid_inputs_old.flatten_inputs(),
    'interpolate': valid_inputs.interpolate_inputs() + valid_inputs_old.interpolate_inputs(),
    'is_nonzero': valid_inputs.is_nonzero_inputs() + valid_inputs_old.is_nonzero_inputs(),
    'layer_norm': valid_inputs.layer_norm_inputs() + valid_inputs_old.layer_norm_inputs(),
    # 'scatter_add': scatter_add_inputs,
    'scatter_add': valid_inputs_old.scatter_add_inputs(),
    'chunk': valid_inputs.chunk_inputs() + valid_inputs_old.chunk_inputs(),
    'isclose': valid_inputs.isclose_inputs() + valid_inputs_old.isclose_inputs(),
    'heaviside': valid_inputs.heaviside_inputs() + valid_inputs_old.heaviside_inputs(),
    'gt': valid_inputs.gt_inputs() + valid_inputs_old.gt_inputs(),
    'hypot': valid_inputs.hypot_inputs() + valid_inputs_old.hypot_inputs(),
    'igamma': valid_inputs.igamma_inputs() + valid_inputs_old.igamma_inputs(),
    'inner': valid_inputs.inner_inputs() + valid_inputs_old.inner_inputs(),
    'lcm': valid_inputs.lcm_inputs() + valid_inputs_old.lcm_inputs(),
    'le': valid_inputs.le_inputs() + valid_inputs_old.le_inputs(),
    'einsum': valid_inputs.einsum_inputs() + valid_inputs_old.einsum_inputs(),
    'index_select': valid_inputs.index_select_inputs() + valid_inputs_old.index_select_inputs(),
    'as_strided': valid_inputs.as_strided_inputs() + valid_inputs_old.as_strided_inputs(),
    'broadcast_to': valid_inputs.broadcast_to_inputs() + valid_inputs_old.broadcast_to_inputs(),
    'cat': valid_inputs.cat_inputs() + valid_inputs_old.cat_inputs(),
    'flip': valid_inputs.flip_inputs() + valid_inputs_old.flip_inputs(),
    'Flatten': valid_inputs.flatten_inputs_2() + valid_inputs_old.flatten_inputs_2(),
    'FractionalMaxPool2d': valid_inputs.fractional_max_pool2d_inputs() + valid_inputs_old.fractional_max_pool2d_inputs(),
    'adaptive_avg_pool2d': valid_inputs.adaptive_avg_pool2d_inputs() + valid_inputs_old.adaptive_avg_pool2d_inputs(),
    'amax': valid_inputs.amax_inputs() + valid_inputs_old.amax_inputs(),
    'atan2': valid_inputs.atan2_inputs() + valid_inputs_old.atan2_inputs(),
    'cosine_similarity': valid_inputs.cosine_similarity_inputs() + valid_inputs_old.cosine_similarity_inputs(),
    'cross': valid_inputs.cross_inputs() + valid_inputs_old.cross_inputs(),
    'cross_entropy': valid_inputs.cross_entropy_inputs(),
    'dist': valid_inputs.dist_inputs() + valid_inputs_old.dist_inputs(),
    'floor_divide': valid_inputs.floor_divide_inputs() + valid_inputs_old.floor_divide_inputs(),
    'fmin': valid_inputs.fmin_inputs() + valid_inputs_old.fmin_inputs(),
    'ger': valid_inputs.ger_inputs() + valid_inputs_old.ger_inputs(),
    'GroupNorm': valid_inputs.groupnorm_inputs() + valid_inputs_old.groupnorm_inputs(),
    'MaxPool3d': valid_inputs.MaxPool3d_inputs() + valid_inputs_old.maxpool3d_inputs(),
    'PixelShuffle': valid_inputs.pixel_shuffle_inputs() + valid_inputs_old.pixel_shuffle_inputs(),
    'adaptive_max_pool2d': valid_inputs.adaptive_max_pool2d_inputs() + valid_inputs_old.adaptive_max_pool2d_inputs(),
    'alpha_dropout': valid_inputs.alpha_dropout_inputs() + valid_inputs_old.alpha_dropout_inputs(),
    'bitwise_and': valid_inputs.bitwise_and_inputs() + valid_inputs_old.bitwise_and_inputs(),
    'LayerNorm': valid_inputs.LayerNorm_inputs() + valid_inputs_old.LayerNorm_inputs(),
    'Linear': valid_inputs.Linear_inputs() + valid_inputs_old.Linear_inputs(),
    'MaxPool2d': valid_inputs.MaxPool2d_inputs() + valid_inputs_old.maxpool2d_inputs(),
    'PReLU_': valid_inputs.PReLU_inputs() + valid_inputs_old.PReLU_inputs(),
    'Softmax': valid_inputs.Softmax_inputs() + valid_inputs.Softmax_inputs_2() + valid_inputs_old.Softmax_inputs(),
    'Softmin': valid_inputs.softmin_inputs() + valid_inputs_old.softmin_inputs(),
    'bincount': valid_inputs.bincount_inputs() + valid_inputs_old.bincount_inputs(),
    'bitwise_or': valid_inputs.bitwise_or_inputs() + valid_inputs_old.bitwise_or_inputs(),
    'bitwise_xor': valid_inputs.bitwise_xor_inputs() + valid_inputs_old.bitwise_xor_inputs(),
    'bmm': valid_inputs.bmm_inputs() + valid_inputs_old.bmm_inputs(),
    'cdist': valid_inputs.cdist_inputs() + valid_inputs_old.cdist_inputs(),
    'complex': valid_inputs.complex_inputs() + valid_inputs_old.complex_inputs(),
    'copysign': valid_inputs.copysign_inputs() + valid_inputs_old.copysign_inputs(),
    'dot': valid_inputs.dot_inputs() + valid_inputs_old.dot_inputs(),
    'eq': valid_inputs.eq_inputs() + valid_inputs_old.eq_inputs(),
    'float_power': valid_inputs.float_power_inputs() + valid_inputs_old.float_power_inputs(),
    'ge': valid_inputs.ge_inputs() + valid_inputs_old.ge_inputs(),
    'allclose': valid_inputs.allclose_inputs() + valid_inputs_old.allclose_inputs(),
    'L1Loss': valid_inputs.l1loss_inputs() + valid_inputs_old.l1loss_inputs(),
    'MSELoss': valid_inputs.MSELoss_inputs() + valid_inputs_old.MSELoss_inputs(),
    'MultiMarginLoss': valid_inputs.multimarginloss_inputs() + valid_inputs_old.multimarginloss_inputs(),
    'PairwiseDistance': valid_inputs.pairwise_distance_inputs() + valid_inputs_old.pairwise_distance_inputs(),
    'PoissonNLLLoss': valid_inputs.poisson_nll_loss_inputs() + valid_inputs_old.poisson_nll_loss_inputs(),
    'addmm': valid_inputs.addmm_inputs() + valid_inputs_old.addmm_inputs(),
    'embedding_bag': valid_inputs.embedding_bag_inputs() + valid_inputs_old.embedding_bag_inputs(),
    'binaryCrossEntropyWithLogits': valid_inputs.binary_cross_entropy_with_logits_inputs() + valid_inputs_old.binary_cross_entropy_with_logits_inputs(),
    'addcdiv': valid_inputs.addcdiv_inputs() + valid_inputs_old.addcdiv_inputs(),
    'addmv': valid_inputs.addmv_inputs() + valid_inputs_old.addmv_inputs(),
    'addr': valid_inputs.addr_inputs() + valid_inputs_old.addr_inputs(),
    'MultiLabelSoftMarginLoss': valid_inputs.multilabel_soft_margin_loss_inputs() + valid_inputs_old.multilabel_soft_margin_loss_inputs(),
    'NLLLoss': valid_inputs.nll_loss_inputs() + valid_inputs_old.nll_loss_inputs(),
    'batch_norm': valid_inputs.batch_norm_inputs() + valid_inputs_old.batch_norm_inputs(),
    'LSTMCell': valid_inputs.lstm_cell_inputs() + valid_inputs_old.lstm_cell_inputs(),
    'MarginRankingLoss': valid_inputs.margin_ranking_loss_inputs() + valid_inputs_old.margin_ranking_loss_inputs(),
    'set_autocast_cpu_enabled': valid_inputs.set_autocast_cpu_enabled_inputs(),
    'bitwise_right_shift': valid_inputs.bitwise_right_shift_inputs(),
    'arcsinh_': valid_inputs.arcsinh_inputs(),
    'set_num_interop_threads': valid_inputs.set_num_interop_threads_inputs(),
    'solve_ex': valid_inputs.solve_ex_inputs(),
    'fake_quantize_per_tensor_affine': valid_inputs.fake_quantize_per_tensor_affine_inputs(),
    'typename': valid_inputs.typename_inputs(),
    'swapaxes': valid_inputs.swapaxes_inputs(),
    'scatter_reduce': valid_inputs.scatter_reduce_inputs(),
    'cross': valid_inputs.cross_inputs(),
    'aminmax': valid_inputs.aminmax_inputs(),
    'argwhere': valid_inputs.argwhere_inputs(),
    'all': valid_inputs.all_inputs(),
    'matrix_power': valid_inputs.matrix_power_inputs(),
    'conj_physical_': valid_inputs.conj_physical__inputs(),
    'miopen_batch_norm': valid_inputs.miopen_batch_norm_inputs(),
    'atan_': valid_inputs.atan_inputs(),
    'LazyInstanceNorm2d': valid_inputs.lazy_instance_norm2d_inputs(),
    'vstack': valid_inputs.vstack_inputs(),
    'lu': valid_inputs.lu_inputs(),
    'clamp_max': valid_inputs.clamp_max_inputs(),
    'cos_': valid_inputs.cos__inputs(),
    'get_num_threads': valid_inputs.get_num_threads_inputs(),
    'softmax': valid_inputs.softmax_inputs(),
    'pinv': valid_inputs.torch_linalg_pinv_inputs(),
    'dsplit': valid_inputs.dsplit_inputs(),
    'CircularPad1d': valid_inputs.circularpad1d_inputs(),
    'Mish': valid_inputs.mish_inputs(),
    'view_as_complex_copy': valid_inputs.view_as_complex_copy_inputs(),
    'hspmm': valid_inputs.hspmm_inputs(),
    'trunc_': valid_inputs.trunc__inputs(),
    'set_autocast_ipu_enabled': valid_inputs.set_autocast_ipu_enabled_inputs(),
    'isreal': valid_inputs.isreal_inputs(),
    'ifftshift': valid_inputs.ifftshift_inputs(),
    'native_dropout': valid_inputs.native_dropout_inputs(),
    'is_grad_enabled': valid_inputs.is_grad_enabled_inputs(),
    'flatten': valid_inputs.flatten_inputs(),
    'arccos_': valid_inputs.arccos__inputs(),
    'index_copy': valid_inputs.index_copy_inputs(),
    'arctan': valid_inputs.arctan_inputs(),
    'less_equal': valid_inputs.less_equal_inputs(),
    'bartlett_window': valid_inputs.bartlett_window_inputs(),
    'reciprocal_': valid_inputs.reciprocal__inputs(),
    'as_strided_copy': valid_inputs.as_strided_copy_inputs(),
    'arccos': valid_inputs.arccos_inputs(),
    'AdaptiveAvgPool3d': valid_inputs.adaptive_avg_pool3d_inputs(),
    'embedding_bag': valid_inputs.embedding_bag_inputs(),
    'moveaxis': valid_inputs.moveaxis_inputs(),
    'tan_': valid_inputs.tan__inputs(),
    'tanh': valid_inputs.tanh_inputs(),
    'Softplus': valid_inputs.Softplus_inputs(),
    'vector_norm': valid_inputs.vector_norm_inputs(),
    'is_autocast_ipu_enabled': valid_inputs.is_autocast_ipu_enabled_inputs(),
    'ZeroPad1d': valid_inputs.ZeroPad1d_inputs(),
    'manual_seed': valid_inputs.manual_seed_inputs(),
    'log_softmax': valid_inputs.log_softmax_inputs() + valid_inputs.log_softmax_inputs_2() + valid_inputs.log_softmax_inputs_3() + valid_inputs.log_softmax_inputs_4(),
    'set_warn_always': valid_inputs.set_warn_always_inputs(),
    'gcd_': valid_inputs.gcd__inputs(),
    'rsub': valid_inputs.rsub_inputs(),
    'ceil_': valid_inputs.ceil__inputs(),
    'ldexp_': valid_inputs.ldexp_inputs(),
    'parameters_to_vector': valid_inputs.parameters_to_vector_inputs(),
    'get_autocast_cpu_dtype': valid_inputs.get_autocast_cpu_dtype_inputs(),
    'swapdims': valid_inputs.swapdims_inputs(),
    'any': valid_inputs.any_inputs(),
    'vsplit': valid_inputs.vsplit_inputs(),
    'set_anomaly_enabled': valid_inputs.set_anomaly_enabled_inputs(),
    'BCEWithLogitsLoss': valid_inputs.BCEWithLogitsLoss_inputs(),
    'autocast_increment_nesting': valid_inputs.autocast_increment_nesting_inputs(),
    'frexp': valid_inputs.frexp_inputs(),
    'eig': valid_inputs.eig_inputs(),
    'arcsin_': valid_inputs.arcsin_inputs(),
    'set_num_threads': valid_inputs.set_num_threads_inputs(),
    'is_tracing': valid_inputs.is_tracing_inputs(),
    'arctanh': valid_inputs.arctanh_inputs(),
    'eigvals': valid_inputs.linalg_eigvals_inputs(),
    'CrossEntropyLoss': valid_inputs.cross_entropy_loss_inputs(),
    'is_anomaly_enabled': valid_inputs.is_anomaly_enabled_inputs(),
    'HuberLoss': valid_inputs.HuberLoss_inputs(),
    'clip': valid_inputs.clip_inputs(),
    'rsqrt_': valid_inputs.rsqrt_inputs(),
    'erfc_': valid_inputs.erfc_inputs(),
    'Parameter': valid_inputs.parameter_inputs(),
    'index_put': valid_inputs.index_put_inputs(),
    'hann_window': valid_inputs.hann_window_inputs(),
    'is_autocast_cache_enabled': valid_inputs.is_autocast_cache_enabled_inputs(),
    'celu_': valid_inputs.celu_inputs() + valid_inputs.celu_inputs_2() + valid_inputs.celu_inputs_3(),
    'ReflectionPad3d': valid_inputs.ReflectionPad3d_inputs(),
    'save': valid_inputs.torch_save_inputs(),
    'PixelUnshuffle': valid_inputs.pixel_unshuffle_inputs(),
    'concatenate': valid_inputs.concatenate_inputs(),
    'cond': valid_inputs.cond_inputs(),
    'corrcoef': valid_inputs.corrcoef_inputs(),
    'quantile': valid_inputs.quantile_inputs(),
    'isin': valid_inputs.isin_inputs(),
    'gammaincc': valid_inputs.gammaincc_inputs(),
    'arcsin': valid_inputs.arcsin_inputs(),
    'CELU': valid_inputs.celu_inputs() + valid_inputs.celu_inputs_2() + valid_inputs.celu_inputs_3(),
    'igammac': valid_inputs.igammac_inputs(),
    'sigmoid_': valid_inputs.sigmoid__inputs(),
    'Sigmoid': valid_inputs.Sigmoid_inputs(),
    'is_autocast_enabled': valid_inputs.is_autocast_enabled_inputs(),
    'scatter_add': valid_inputs.scatter_add_inputs(),
    'row_stack': valid_inputs.row_stack_inputs(),
    'negative': valid_inputs.negative_inputs(),
    'CosineSimilarity': valid_inputs.cosine_similarity_inputs(),
    'view_as_real': valid_inputs.view_as_real_inputs(),
    'qr': valid_inputs.qr_inputs(),
    'cholesky': valid_inputs.cholesky_inputs(),
    'GaussianNLLLoss': valid_inputs.gaussian_nllloss_inputs(),
    'acosh_': valid_inputs.acosh_inputs(),
    'is_autocast_cpu_enabled': valid_inputs.is_autocast_cpu_enabled_inputs(),
    'sin_': valid_inputs.sin__inputs(),
    'divide': valid_inputs.generate_divide_inputs(),
    'prepare_multiprocessing_environment': valid_inputs.prepare_multiprocessing_environment_inputs(),
    'get_rng_state': valid_inputs.get_rng_state_inputs(),
    'set_autocast_enabled': valid_inputs.set_autocast_enabled_inputs(),
    'square_': valid_inputs.square__inputs(),
    'arccosh': valid_inputs.arccosh_inputs(),
    'count_nonzero': valid_inputs.count_nonzero_inputs(),
    'index_select': valid_inputs.index_select_inputs(),
    'expit': valid_inputs.expit_inputs(),
    'set_deterministic_debug_mode': valid_inputs.set_deterministic_debug_mode_inputs(),
    'select': valid_inputs.torch_select_inputs(),
    'Tanhshrink': valid_inputs.Tanhshrink_inputs(),
    'get_default_device': valid_inputs.get_default_device_inputs(),
    'matrix_exp': valid_inputs.matrix_exp_inputs(),
    'histogram': valid_inputs.torch_histogram_inputs(),
    'Tanh': valid_inputs.Tanh_inputs(),
    'is_inference_mode_enabled': valid_inputs.is_inference_mode_enabled_inputs(),
    'native_channel_shuffle': valid_inputs.native_channel_shuffle_inputs(),
    'equal': valid_inputs.equal_inputs(),
    'tile': valid_inputs.torch_tile_inputs(),
    'ldexp': valid_inputs.ldexp_inputs(),
    'crow_indices_copy': valid_inputs.crow_indices_copy_inputs(),
    'arctan_': valid_inputs.arctan_inputs(),
    'SoftMarginLoss': valid_inputs.soft_margin_loss_inputs(),
    'randn_like': valid_inputs.randn_like_inputs(),
    'subtract': valid_inputs.subtract_inputs(),
    'conj_physical': valid_inputs.conj_physical_inputs(),
    'column_stack': valid_inputs.column_stack_inputs(),
    'true_divide': valid_inputs.true_divide_inputs(),
    'negative_': valid_inputs.negative_inputs(),
    'greater_equal': valid_inputs.greater_equal_inputs(),
    'empty': valid_inputs.torch_empty_inputs(),
    'TripletMarginLoss': valid_inputs.triplet_margin_loss_inputs(),
    'alias_copy': valid_inputs.alias_copy_inputs(),
    'entr': valid_inputs.entr_inputs(),
    'asinh_': valid_inputs.asinh_inputs(),
    'log_': valid_inputs.log_inputs(),
    'set_autocast_xla_enabled': valid_inputs.set_autocast_xla_enabled_inputs(),
    'vecdot': valid_inputs.vecdot_inputs(),
    'ScriptWarning': valid_inputs.script_warning_inputs(),
    'Unflatten': valid_inputs.unflatten_inputs(),
    'is_floating_point': valid_inputs.is_floating_point_inputs(),
    'LocalResponseNorm': valid_inputs.local_response_norm_inputs(),
    'slogdet': valid_inputs.slogdet_inputs(),
    'is_storage': valid_inputs.is_storage_inputs(),
    'masked_scatter': valid_inputs.masked_scatter_inputs(),
    'LogSoftmax': valid_inputs.log_softmax_inputs() + valid_inputs.log_softmax_inputs_2() + valid_inputs.log_softmax_inputs_3() + valid_inputs.log_softmax_inputs_4(),
    'CosineEmbeddingLoss': valid_inputs.CosineEmbeddingLoss_inputs(),
    'concat': valid_inputs.concat_inputs(),
    'set_autocast_cache_enabled': valid_inputs.set_autocast_cache_enabled_inputs(),
    'arcsinh': valid_inputs.arcsinh_inputs(),
    'rfftfreq': valid_inputs.rfftfreq_inputs(),
    'atanh_': valid_inputs.atanh_inputs(),
    'unravel_index': valid_inputs.unravel_index_inputs(),
    'lu_unpack': valid_inputs.lu_unpack_inputs(),
    'clone': valid_inputs.clone_inputs(),
    'set_default_device': valid_inputs.set_default_device_inputs(),
    'permute': valid_inputs.permute_inputs(),
    'absolute': valid_inputs.absolute_inputs(),
    'threshold': valid_inputs.threshold_inputs(),
    'cosh_': valid_inputs.cosh__inputs(),
    'get_device': valid_inputs.get_device_inputs(),
    'relu_': valid_inputs.relu_inputs(),
    'fftfreq': valid_inputs.fftfreq_inputs(),
    'quantize_per_tensor': valid_inputs.quantize_per_tensor_inputs(),
    'KLDivLoss': valid_inputs.KLDivLoss_inputs(),
    'clear_autocast_cache': valid_inputs.clear_autocast_cache_inputs(),
    'sqrt_': valid_inputs.sqrt__inputs(),
    'asin_': valid_inputs.asin__inputs(),
    'vander': valid_inputs.vander_inputs(),
    'fftn': valid_inputs.fftn_inputs(),
    'are_deterministic_algorithms_enabled': valid_inputs.are_deterministic_algorithms_enabled_inputs(),
    'cosine_similarity': valid_inputs.cosine_similarity_inputs(),
    'get_default_dtype': valid_inputs.get_default_dtype_inputs(),
    'view_as_complex': valid_inputs.view_as_complex_inputs(),
    'gather': valid_inputs.gather_inputs(),
    'det': valid_inputs.det_inputs(),
    'FractionalMaxPool3d': valid_inputs.fractionalmaxpool3d_inputs(),
    'strict_fusion': valid_inputs.strict_fusion_inputs(),
    'stft': valid_inputs.torch_stft_inputs(),
    'acos_': valid_inputs.acos_inputs(),
    'spherical_bessel_j0': valid_inputs.spherical_bessel_j0_inputs(),
    'bilinear': valid_inputs.bilinear_inputs(),
    'nanmean': valid_inputs.nanmean_inputs(),
    'hamming_window': valid_inputs.hamming_window_inputs(),
    'greater': valid_inputs.greater_inputs(),
    'erf_': valid_inputs.erf__inputs(),
    'ZeroPad2d': valid_inputs.ZeroPad2d_inputs(),
    'vdot': valid_inputs.vdot_inputs(),
    'is_warn_always_enabled': valid_inputs.is_warn_always_enabled_inputs(),
    'is_anomaly_check_nan_enabled': valid_inputs.is_anomaly_check_nan_enabled_inputs(),
    'ones_like': valid_inputs.ones_like_inputs(),
    'not_equal': valid_inputs.not_equal_inputs(),
    'ReLU': valid_inputs.ReLU_inputs(),
    'floor_': valid_inputs.floor_inputs(),
    'narrow_copy': valid_inputs.narrow_copy_inputs(),
    'use_deterministic_algorithms': valid_inputs.use_deterministic_algorithms_inputs(),
    'Threshold': valid_inputs.Threshold_inputs(),
    'erfcx': valid_inputs.erfcx_inputs(),
    'multiply': valid_inputs.multiply_inputs(),
    'logit_': valid_inputs.logit__inputs(),
    'solve_triangular': valid_inputs.solve_triangular_inputs(),
    'is_autocast_xla_enabled': valid_inputs.is_autocast_xla_enabled_inputs(),
    'rfft': valid_inputs.rfft_inputs(),
    'ConstantPad1d': valid_inputs.constant_pad1d_inputs(),
    'less': valid_inputs.less_inputs(),
    'Hardsigmoid': valid_inputs.hardsigmoid_inputs(),
    'diff': valid_inputs.diff_inputs(),
    'hsplit': valid_inputs.hsplit_inputs(),
    'is_scripting': valid_inputs.is_scripting_inputs(),
    'set_grad_enabled': valid_inputs.set_grad_enabled_inputs(),
    'rfft2': valid_inputs.rfft2_inputs(),
    'arctanh_': valid_inputs.arctanh_inputs(),
    'select_copy': valid_inputs.select_copy_inputs(),
    'Error': valid_inputs.jit_error_inputs(),
    'meshgrid': valid_inputs.meshgrid_inputs(),
    'layer_norm': valid_inputs.layer_norm_inputs(),
    'inv': valid_inputs.linalg_inv_inputs(),
    'Identity': valid_inputs.identity_inputs(),
    'nan_to_num': valid_inputs.nan_to_num_inputs(),
    'adjoint': valid_inputs.adjoint_inputs(),
    'clip_': valid_inputs.clip_inputs(),
    'set_module': valid_inputs.set_module_inputs(),
    'addmv_': valid_inputs.addmv_inputs(),
    'diagonal': valid_inputs.diagonal_inputs(),
    'bitwise_left_shift': valid_inputs.bitwise_left_shift_inputs(),
    'fft': valid_inputs.fft_inputs(),
    'eigh': valid_inputs.torch_linalg_eigh_inputs(),
    'fix': valid_inputs.fix_inputs(),
    'SmoothL1Loss': valid_inputs.smoothl1loss_inputs(),
    'LazyInstanceNorm1d': valid_inputs.lazy_instance_norm1d_inputs(),
    'matrix_rank': valid_inputs.matrix_rank_inputs(),
    'lerp': valid_inputs.lerp_inputs(),
    'celu': valid_inputs.celu_inputs() + valid_inputs.celu_inputs_2() + valid_inputs.celu_inputs_3(),
    'fft2': valid_inputs.fft2_inputs(),
    'UninitializedParameter': valid_inputs.uninitialized_parameter_inputs(),
    'take': valid_inputs.take_inputs(),
    'fmax': valid_inputs.fmax_inputs(),
    'fftshift': valid_inputs.fftshift_inputs(),
    'positive': valid_inputs.positive_inputs(),
    'multi_dot': valid_inputs.multi_dot_inputs(),
    'svdvals': valid_inputs.svdvals_inputs(),
    'abs_': valid_inputs.abs__inputs(),
    'multigammaln': valid_inputs.multigammaln_inputs(),
    'get_deterministic_debug_mode': valid_inputs.get_deterministic_debug_mode_inputs(),
    'get_total_norm': valid_inputs.get_total_norm_inputs(),
    'multinomial': valid_inputs.multinomial_inputs(),
    'softmax_': valid_inputs.softmax_inputs(),
    'softmin_': valid_inputs.softmin_inputs(),
    'matrix_exp': valid_inputs.matrix_exp_inputs(),
    'matrix_power': valid_inputs.matrix_power_inputs(),
    'logdet': valid_inputs.logdet_inputs(),
    'logSoftmaxClass': valid_inputs.log_softmax_inputs() + valid_inputs.log_softmax_inputs_2() + valid_inputs.log_softmax_inputs_3() + valid_inputs.log_softmax_inputs_4(),
    'slogdet': valid_inputs.slogdet_inputs(),
    'symeig': valid_inputs.symeig_inputs(),
    'remainder': valid_inputs.remainder_inputs(),
    'lt': valid_inputs.lt_inputs(),
    'logaddexp2': valid_inputs.logaddexp2_inputs(),
    'logaddexp': valid_inputs.logaddexp_inputs(),
    'maximum': valid_inputs.maximum_inputs(),
    'minimum': valid_inputs.minimum_inputs(),
    'nextafter': valid_inputs.nextafter_inputs(),
    'inverse': valid_inputs.inverse_inputs(),
    'chain_matmul': valid_inputs.chain_matmul_inputs(),
    'cumsum': valid_inputs.cumsum_inputs(),
    'avg_pool2d': valid_inputs.avg_pool2d_inputs(),
    'cumprod': valid_inputs.cumprod_inputs(),
    'det': valid_inputs.det_inputs(),
    'amin': valid_inputs.torch_amin_inputs(),
    'argmax': valid_inputs.argmax_inputs(),
    'argmin': valid_inputs.argmin_inputs(),
    'argsort': valid_inputs.argsort_inputs(),
    'unique_consecutive': valid_inputs.unique_consecutive_inputs(),
    'logsumexp': valid_inputs.logsumexp_inputs(),
    'max': valid_inputs.generate_max_inputs(),
    'prod': valid_inputs.prod_inputs(),
    'squeeze': valid_inputs.squeeze_inputs(),
    'log_softmax': valid_inputs.log_softmax_inputs() + valid_inputs.log_softmax_inputs_2() + valid_inputs.log_softmax_inputs_3() + valid_inputs.log_softmax_inputs_4(),
    'pairwise_distance': valid_inputs.pairwise_distance_inputs(),
    'mul': valid_inputs.mul_inputs(),
    'ne': valid_inputs.ne_inputs(),
    'eig': valid_inputs.eig_inputs(),
    'grucell': valid_inputs.GRUCell_inputs(),
    'add': valid_inputs.add_inputs(),
    'stack': valid_inputs.stack_inputs(),
    'unbind': valid_inputs.unbind_inputs(),
    'logical_or': valid_inputs.logical_or_inputs(),
    'masked_select': valid_inputs.masked_select_inputs(),
    'logical_and': valid_inputs.logical_and_inputs(),
    'logical_xor': valid_inputs.logical_xor_inputs(),
    'cartesian_prod': valid_inputs.cartesian_prod_inputs(),
    'unique': valid_inputs.unique_inputs(),
    'DoubleStorage': valid_inputs.DoubleStorage_inputs(),
    'ShortStorage': valid_inputs.ShortStorage_inputs(),
    'enable_grad': valid_inputs.enable_grad_inputs(),
    'HingeEmbeddingLoss': valid_inputs.hinge_embedding_loss_inputs(),
    'asarray': valid_inputs.asarray_inputs(),
    'sspaddmm': valid_inputs.sspaddmm_inputs(),
    'vitals_enabled': valid_inputs.vitals_enabled_inputs(),  # llm end
}

def get_inputs(api, lib="torch", time_budget=30, min_val_inp=5, seed=42):
    api_signature = get_signatures()[api]
    # Return human written inputs if available
    if api in inputs_per_api:
        return augment_inputs(inputs_per_api[api], api_signature)
    
    # Generate valid inputs through random generation otherwise
    input_file = os.path.join(get_dir_in_root(f"valid_inputs_{lib}"), f"{api}.pkl")
    
    list_of_inputs = []
    
    # If there already is a saved file, read from that and concretize
    if os.path.isfile(input_file):
        abstract_inputs = read_pkl(input_file)
        for abs_inp, saved_seed in abstract_inputs:
            rng = np.random.default_rng(saved_seed)
            list_of_inputs.append(concretize_input(abs_inp, api_signature, rng))
    else:   # Generate and save otherwise
        api_driver = get_driver(api, lib=lib)
        valid = 0
        invalid = 0
        abstract_inputs = []
        
        start_time = time.time()
        while (time.time() - start_time < time_budget) and (valid < min_val_inp):
            rng = np.random.default_rng(seed)
            input_dict = get_random_input(api_signature, rng)
            status, exception_message = oracle_crash(api_driver, input_dict, cpu=True)
            if status == "invalid":
                invalid += 1
            else:
                valid += 1
                # Only adding valid inputs
                list_of_inputs.append(input_dict)
                abs_inp = get_abstract_input(input_dict, api_signature)
                # Save the abstract input along with the seed
                abstract_inputs.append((abs_inp, seed))
            
            seed += 1
        
        # Save abstract inputs to file
        save_to_new_pkl(input_file, abstract_inputs)
    
    return augment_inputs(list_of_inputs, api_signature)

def main():
    apis = set()
    total_inputs = 0
    for api, inputs in inputs_per_api.items():
        apis.add(api)
        total_inputs += len(get_inputs(api))
    
    print(f"\n{len(apis)} apis has pre-defined inputs, {round(total_inputs/len(apis), 2)} inputs on average")
    
if __name__ == "__main__":
    main()