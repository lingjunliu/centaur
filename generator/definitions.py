import numpy as np
from learner.invariant_inference import infer_invariants
from learner.inputs import get_inputs
from utils.defaults import *
# TODO: Move the definitions to JSON

############### api definitions ################

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
    "ruleset":  infer_invariants("scatter", get_inputs("scatter")),
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
                        "dim": "integer"
                    },
    "ruleset":  set([
                        ('rule_2', 'input', 'dim')                        
                    ]),
    "random_candidate": {
                            "input": np.random.rand(2,4,343,10,1).astype(np.float32),
                            "dim": 10
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
    "ruleset":  infer_invariants("conv_transpose2d", get_inputs("conv_transpose2d")),
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
                    ('rule_1', 'input', 'other')
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
                    ('rule_2', 'input', 'dim')
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
                    ('rule_3', 'input', 'other')
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
                    ('rule_4', 'input', 'other')
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
                    ('rule_5', 'input', 'index')
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
                    ('rule_6', 'input', 'other')
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

############### map definitions to rules/apis ################

map_defs = {
    "rule_1": r1_definition,
    "rule_2": r2_definition,
    "rule_3": r3_definition,
    "rule_4": r4_definition,
    "rule_5": r5_definition,
    "rule_6": r6_definition,
    "scatter": scatter_definition,
    "atan2": atan2_definition,
    "matmul": matmul_definition,
    "argmin": argmin_definition,
    "conv_transpose2d": conv_transpose2d_definition
}