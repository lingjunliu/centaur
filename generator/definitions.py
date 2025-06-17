import numpy as np
from learner.invariant_inference import infer_invariants
from utils.defaults import *
from utils.new_api_utils import get_signature, get_lib_version

############### api definition examples ################

# Scatter

scatter_definition = {
    "signature":    {
                        "input": "tensor",
                        "dim": "integer",
                        "index": "tensor",
                        "src": "tensor"
                    },
    # Set manually
    # "ruleset":  set([
    #                     ('rule_2', 'input', 'dim'),
    #                     ('rule_2', 'index', 'dim'),
    #                     ('rule_2', 'src', 'dim'),
    #                     ('rule_3', 'input', 'src'),
    #                     ('rule_3', 'input', 'index'),
    #                     ('rule_3', 'src', 'index'),
    #                     ('rule_4', 'input', 'src'),
    #                     ('rule_5', 'input', 'index')
    #                 ]),
    # Using invariant inference
    "ruleset":  infer_invariants("scatter")[0],
    # Easy
    # "random_candidate": {
    #                         "input": np.random.rand(2,4).astype(np.float32),
    #                         "dim": 7,
    #                         "index": np.random.rand(2,1).astype(np.int64),
    #                         "src": np.random.rand(2,1,3).astype(np.float32)
    #                     },
    # Hard
    "random_candidate": {
                            "input": np.random.rand(2,4, 343, 10).astype(np.float32),
                            "dim": 7,
                            "index": np.random.rand(2).astype(np.int64),
                            "src": np.random.rand(2,1).astype(np.complex64)
                        },
    "arg_order": ['input', 'dim', 'index', 'src'],
    # set limits based on index of argument in the order
    # three lists per argument: value list, dtype list, range list
    # min_num, max_num, min_size, max_size
    "limits":   [            
                    domain_limits['tensor'], # input value
                    domain_limits['tensor_dtype'],  # input dtype
                    domain_limits['tensor_value_range'],    # input range of values
                    domain_limits['integer'], # dim value
                    domain_limits['integer_dtype'], # dim dtype, only allow integer types
                    domain_limits['integer_value_range'],    # dim, range of values not really needed, added for symmetry
                    domain_limits['tensor'], # index value
                    domain_limits['tensor_dtype'],  # index dtype
                    domain_limits['tensor_value_range'],    # index range of values
                    domain_limits['tensor'], # src value
                    domain_limits['tensor_dtype'],  # src dtype
                    domain_limits['tensor_value_range'],    # src range of values
                ]
}

# atan2

atan2_definition = {
    "signature":    {
                        "input": "tensor",
                        "other": "tensor"
                    },
    "ruleset":  set([
                        ('rule_3', 'input', 'other'),
                        ('rule_6', 'input', 'other'),
                        ('rule_1', 'input', 'other'),
                        ('rule_4', 'input', 'other'),
                        ('rule_6', 'other', 'input')
                    ]),
    "random_candidate": {
                            "input": np.random.rand(2,4,343,10,1).astype(np.float32),
                            "other": np.random.rand(2,1,3).astype(np.int64)
                        },
    "arg_order": ['input', 'other'],
    # set limits based on index of argument in the order
    # three lists per argument: value list, dtype list, range list
    # min_num, max_num, min_size, max_size
    "limits":   [            
                    domain_limits['tensor'], # input value
                    domain_limits['tensor_dtype'],  # input dtype
                    domain_limits['tensor_value_range'],    # input range of values
                    domain_limits['tensor'], # other value
                    domain_limits['tensor_dtype'],  # other dtype
                    domain_limits['tensor_value_range'],    # other range of values
                ]
}

# matmul

matmul_definition = {
    "signature":    {
                        "input": "tensor",
                        "other": "tensor"
                    },
    "ruleset":  set([
                        ('rule_6', 'input', 'other'),
                        ('rule_4', 'input', 'other')                        
                    ]),
    "random_candidate": {
                            "input": np.random.rand(2,4,343,10,1).astype(np.float32),
                            "other": np.random.rand(2,1,3).astype(np.int64)
                        },
    "arg_order": ['input', 'other'],
    # set limits based on index of argument in the order
    # three lists per argument: value list, dtype list, range list
    # min_num, max_num, min_size, max_size
    "limits":   [            
                    domain_limits['tensor'], # input value
                    domain_limits['tensor_dtype'],  # input dtype
                    domain_limits['tensor_value_range'],    # input range of values
                    domain_limits['tensor'], # other value
                    domain_limits['tensor_dtype'],  # other dtype
                    domain_limits['tensor_value_range'],    # other range of values
                ]
}

# argmin

argmin_definition = {
    "signature":    {
                        "input": "tensor",
                        "dim": "integer",
                    },
    "ruleset":  set([
                        ('rule_2', 'input', 'dim')                        
                    ]),
    "random_candidate": {
                            "input": np.random.rand(2,4,343,10,1).astype(np.float32),
                            "dim": 10,
                            "keepdim": True
                        },
    "arg_order": ['input', 'dim'],
    # set limits based on index of argument in the order
    # three lists per argument: value list, dtype list, range list
    # min_num, max_num, min_size, max_size
    "limits":   [            
                    domain_limits['tensor'], # input value
                    domain_limits['tensor_dtype'],  # input dtype
                    domain_limits['tensor_value_range'],    # input range of values
                    domain_limits['integer'], # dim value
                    domain_limits['integer_dtype'],  # dim dtype
                    domain_limits['integer_value_range'],    # dim range of values
                ]
}

# conv_transpose2d

conv_transpose2d_definition = {
    "signature":    {
                        "input": "tensor",
                        "weight": "tensor",
                        "stride": "integer",
                        "padding": "integer"
                    },
    # Set manually
    # "ruleset":  set([
    #                     ('rule_3', 'input', 'weight'),
    #                     ('rule_4', 'input', 'weight'),
    #                     ('rule_7', 'weight', 'input')
    #                 ]),
    # Using invariant inference
    "ruleset":  infer_invariants("conv_transpose2d")[0],
    "random_candidate": {
                            "input": np.random.rand(2,4,343,10,1).astype(np.float32),
                            "weight": np.random.rand(2,343,1).astype(np.float32),
                            "stride": 10,
                            "padding": 1
                        },
    "arg_order": ['input', 'weight', 'stride', 'padding'],
    # set limits based on index of argument in the order
    # three lists per argument: value list, dtype list, range list
    # min_num, max_num, min_size, max_size
    "limits":   [            
                    domain_limits['tensor'], # input value
                    domain_limits['tensor_dtype'],  # input dtype
                    domain_limits['tensor_value_range'],    # input range of values
                    domain_limits['tensor'], # weight value
                    domain_limits['tensor_dtype'],  # weight dtype
                    domain_limits['tensor_value_range'],    # weight range of values
                    domain_limits['integer'], # stride value
                    domain_limits['integer_dtype'],  # stride dtype
                    domain_limits['integer_value_range'],    # stride range of values
                    domain_limits['integer'], # padding value
                    domain_limits['integer_dtype'],  # padding dtype
                    domain_limits['integer_value_range'],    # padding range of values
                ]
}


############### example definitions ################
# For testing single rules

# define rule 1
r1_definition = {
    "signature":    {
                        "input": "tensor",
                        "other": "tensor"
                    },
    "ruleset":  set([
                    (2, 'rule_1', 'input', 'other')
                ]),
    "random_candidate": {
                            "input": np.random.rand(3,2,5,9,343,2),
                            "other": np.random.rand(4, 2)
                        },
    "arg_order": ['input', 'other'],
    # set limits based on index of argument in the order
    # three lists per argument: value list, dtype list, range list
    # min_num, max_num, min_size, max_size
    "limits":   [            
                    domain_limits['tensor'], # input value
                    domain_limits['tensor_dtype'],  # input dtype
                    domain_limits['tensor_value_range'], # input value range
                    domain_limits['tensor'], # other value
                    domain_limits['tensor_dtype'],  # other dtype
                    domain_limits['tensor_value_range'] # other value range
                ]
}

# define rule 2
r2_definition = {
    "signature":    {
                        "input": "tensor",
                        "dim": "integer"
                    },
    "ruleset":  set([
                    (2, 'rule_2', 'input', 'dim')
                ]),
    "random_candidate": {
                            "input": np.random.rand(3,2,5,9,343,2),
                            "dim": 25
                        },
    "arg_order": ['input', 'dim'],
    # set limits based on index of argument in the order
    # three lists per argument: value list, dtype list, range list
    # min_num, max_num, min_size, max_size
    "limits":   [            
                    domain_limits['tensor'], # input value
                    domain_limits['tensor_dtype'],  # input dtype
                    domain_limits['tensor_value_range'], # input value range
                    domain_limits['integer'], # dim value
                    domain_limits['integer_dtype'],  # dim dtype, integer only
                    domain_limits['integer_value_range']    # dim value range, we don't use it for integers, it is here just for cohesion
                ]
}

# define rule 3
r3_definition = {
    "signature":    {
                        "input": "tensor",
                        "other": "tensor"
                    },
    "ruleset":  set([
                    (2, 'rule_3', 'input', 'other')
                ]),
    "random_candidate": {
                            "input": np.random.rand(3,2,5,9,343,2),
                            "other": np.random.rand(4, 2)
                        },
    "arg_order": ['input', 'other'],
    # set limits based on index of argument in the order
    # three lists per argument: value list, dtype list, range list
    # min_num, max_num, min_size, max_size
    "limits":   [            
                    domain_limits['tensor'], # input value
                    domain_limits['tensor_dtype'],  # input dtype
                    domain_limits['tensor_value_range'], # input value range
                    domain_limits['tensor'], # other value
                    domain_limits['tensor_dtype'],  # other dtype
                    domain_limits['tensor_value_range'] # other value range
                ]
}

# define rule 4
r4_definition = {
    "signature":    {
                        "input": "tensor",
                        "other": "tensor"
                    },
    "ruleset":  set([
                    (2, 'rule_4', 'input', 'other')
                ]),
    "random_candidate": {
                            "input": np.random.rand(3,2,5,9,343,2).astype(np.int32),
                            "other": np.random.rand(4, 2).astype(np.complex128)
                        },
    "arg_order": ['input', 'other'],
    # set limits based on index of argument in the order
    # three lists per argument: value list, dtype list, range list
    # min_num, max_num, min_size, max_size
    "limits":   [            
                    domain_limits['tensor'], # input value
                    domain_limits['tensor_dtype'],  # input dtype
                    domain_limits['tensor_value_range'], # input value range
                    domain_limits['tensor'], # other value
                    domain_limits['tensor_dtype'],  # other dtype
                    domain_limits['tensor_value_range'] # other value range
                ]
}

# define rule 5
r5_definition = {
    "signature":    {
                        "input": "tensor",
                        "index": "tensor"
                    },
    "ruleset":  set([
                    (2, 'rule_5', 'input', 'index')
                ]),
    "random_candidate": {
                            "input": np.random.rand(3,2,5,9,343,2),
                            "index": np.random.rand(4, 2).astype(np.int64)
                            # Fixing the dtype for now, we might need a uniary invariant to make sure this stays int or update rule 5 to require index tensors to be of int types
                        },
    "arg_order": ['input', 'index'],
    # set limits based on index of argument in the order
    # three lists per argument: value list, dtype list, range list
    # min_num, max_num, min_size, max_size
    "limits":   [            
                    domain_limits['tensor'], # input value
                    domain_limits['tensor_dtype'],  # input dtype
                    domain_limits['tensor_value_range'], # input value range
                    domain_limits['tensor'], # index value
                    domain_limits['tensor_dtype'],  # index dtype
                    domain_limits['tensor_value_range'] # index value range
                ]
}

# define rule 6
r6_definition = {
    "signature":    {
                        "input": "tensor",
                        "other": "tensor"
                    },
    "ruleset":  set([
                    (2, 'rule_6', 'input', 'other')
                ]),
    "random_candidate": {
                            "input": np.random.rand(3,2,5,9,343,2),
                            "other": np.random.rand(4, 2)
                        },
    "arg_order": ['input', 'other'],
    # set limits based on index of argument in the order
    # three lists per argument: value list, dtype list, range list
    # min_num, max_num, min_size, max_size
    "limits":   [            
                    domain_limits['tensor'], # input value
                    domain_limits['tensor_dtype'],  # input dtype
                    domain_limits['tensor_value_range'], # input value range
                    domain_limits['tensor'], # other value
                    domain_limits['tensor_dtype'],  # other dtype
                    domain_limits['tensor_value_range'] # other value range
                ]
}

# define rule 11
r11_definition = {
    "signature":    {
                        "input": "tensor",
                        "dim": "integer",
                        "index": "tensor"
                    },
    "ruleset":  set([
                    (3, 'rule_11', 'input', 'dim', 'index')
                ]),
    "random_candidate": {
                            "input": np.random.rand(3,2,5,9,343,2),
                            "dim": 10,
                            "index": np.random.rand(4, 2)
                        },
    "arg_order": ['input', 'dim', 'index'],
    # set limits based on index of argument in the order
    # three lists per argument: value list, dtype list, range list
    # min_num, max_num, min_size, max_size
    "limits":   [            
                    domain_limits['tensor'], # input value
                    domain_limits['tensor_dtype'],  # input dtype
                    domain_limits['tensor_value_range'], # input value range
                    domain_limits['integer'], # dim value
                    domain_limits['integer_dtype'],  # dim dtype, integer only
                    domain_limits['integer_value_range'],    # dim value range, we don't use it for integers, it is here just for cohesion
                    domain_limits['tensor'], # index value
                    domain_limits['tensor_dtype'],  # index dtype
                    domain_limits['tensor_value_range'] # index value range
                ]
}

############### map definitions to rules/apis ################

map_defs = {
    "rule_1": r1_definition,
    "rule_2": r2_definition,
    "rule_3": r3_definition,
    "rule_4": r4_definition,
    "rule_5": r5_definition,
    "rule_6": r6_definition,
    "rule_11": r11_definition,
    "scatter": scatter_definition,
    "atan2": atan2_definition,
    "matmul": matmul_definition,
    "argmin": argmin_definition,
    "conv_transpose2d": conv_transpose2d_definition
}

############### get definitions per api ################

'''
    Get definition per API with an empty random candidate
'''
def get_definition(api, z3=False, lib="torch", suffix=0):
    signature = get_signature(api, lib=lib, suffix=suffix)
    
    definition = {
        "api": api,
        "signature": signature,
        "ruleset":  infer_invariants(api, z3=z3, lib=lib, suffix=suffix)[0],
        "random_candidate": {},
        "arg_order": list(signature.keys()),
        "limits": [],
        "suffix": suffix
    }
    
    definition["random_candidate"] = {}
    
    for arg, domain in signature.items():
        if domain == "tensor_list":
            domain = "tensor"   # hack until tensor_list is supported
        
        # Limits
        definition["limits"].append(domain_limits[domain])
        definition["limits"].append(domain_limits[f'{domain}_dtype'])
        definition["limits"].append(domain_limits[f'{domain}_value_range'])
    
    return definition
