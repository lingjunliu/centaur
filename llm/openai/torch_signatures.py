signatures = {}


















signatures["torch.special.xlog1py_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.xlog1py_2"] = {
    "args": {
        "input": "tensor",
        "other": "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.xlog1py_3"] = {
    "args": {
        "input": "tensor",
        "other": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.xlog1py_4"] = {
    "args": {
        "input": "float",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.xlog1py_5"] = {
    "args": {
        "input": "integer",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.floor_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.init.sparse_"] = {
    "args": {
        "tensor": "tensor",
        "sparsity": "float"
    },
    "kwargs": {
        "std": "float"
    },
    "inner": {},
}
signatures["torch.nansum_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.nansum_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.nansum_3"] = {
    "args": {
        "input": "tensor",
        "dim": "tuple"
    },
    "kwargs": {
        "keepdim": "boolean",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.nansum_4"] = {
    "args": {
        "input": "tensor",
        "dim": "list"  # list of integers
    },
    "kwargs": {
        "keepdim": "boolean",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.sqrt_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.MaxPool3d_1"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.MaxPool3d_2"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.MaxPool3d_3"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "tuple",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.MaxPool3d_4"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "tuple",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.MaxPool3d_5"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "integer",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.MaxPool3d_6"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "integer",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.MaxPool3d_7"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.MaxPool3d_8"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.MaxPool3d_9"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.MaxPool3d_10"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.MaxPool3d_11"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "tuple",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.MaxPool3d_12"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "tuple",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.MaxPool3d_13"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "integer",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.MaxPool3d_14"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "integer",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.MaxPool3d_15"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.MaxPool3d_16"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.take"] = {
    "args": {
        "input": "tensor",
        "index": "tensor",  # LongTensor
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.arccosh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.multiply_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.multiply_2"] = {
    "args": {
        "input": "tensor",
        "other": "float"  # number; could also be complex in PyTorch
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.multiply_3"] = {
    "args": {
        "input": "tensor",
        "other": "integer"  # number; could also be complex in PyTorch
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.isposinf"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.acos_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.reshape"] = {
    "args": {
        "input": "tensor",
        "shape": "tuple"  # could also accept list-like in practice
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.ndtri"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.set_autocast_enabled"] = {
    "args": {
        "enabled": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.logsigmoid"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.get_rng_state"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.absolute"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.abs"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # optional
    },
    "inner": {},
}
signatures["torch.addbmm_1"] = {
    "args": {
        "input": "tensor",
        "batch1": "tensor",
        "batch2": "tensor"
    },
    "kwargs": {
        "beta": "float",  # Number; can also be integer depending on dtype
        "alpha": "float", # Number; can also be integer depending on dtype
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.addbmm_2"] = {
    "args": {
        "input": "tensor",
        "batch1": "tensor",
        "batch2": "tensor"
    },
    "kwargs": {
        "beta": "integer",
        "alpha": "integer",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.adjoint"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.allclose"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "rtol": "float",
        "atol": "float",
        "equal_nan": "boolean"
    },
    "inner": {},
}
signatures["torch.all_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}

signatures["torch.all_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}

signatures["torch.all_3"] = {
    "args": {
        "input": "tensor",
        "dim": "tuple"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.angle"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.any_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}

signatures["torch.any_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}

signatures["torch.any_3"] = {
    "args": {
        "input": "tensor",
        "dim": "tuple"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.arccos"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.arcsinh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.arcsin"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.arctanh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.arctan"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.argmax_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.argmax_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean"
    },
    "inner": {},
}
signatures["torch.argmin"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer",  # could be None when omitted
        "keepdim": "boolean"
    },
    "inner": {}
}
signatures["torch.argsort"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "descending": "boolean"
    },
    "kwargs": {
        "stable": "boolean"
    },
    "inner": {},
}
signatures["torch.argwhere"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.bitwise_and_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.bitwise_and_2"] = {
    "args": {
        "input": "tensor",
        "other": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.bitwise_and_3"] = {
    "args": {
        "input": "tensor",
        "other": "boolean"  # scalar bool supported when input is bool dtype
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.bitwise_left_shift_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.bitwise_left_shift_2"] = {
    "args": {
        "input": "tensor",
        "other": "integer"  # could also accept Python ints; shift counts should be integers
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.bitwise_or_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.bitwise_or_2"] = {
    "args": {
        "input": "tensor",
        "other": "integer"  # could also be a Python bool, see _3
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.bitwise_or_3"] = {
    "args": {
        "input": "tensor",
        "other": "boolean"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.bitwise_right_shift_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.bitwise_right_shift_2"] = {
    "args": {
        "input": "tensor",
        "other": "integer"  # bool likely accepted too, but using integer
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.bitwise_right_shift_3"] = {
    "args": {
        "input": "integer",  # bool likely accepted too, but using integer
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.bitwise_right_shift_4"] = {
    "args": {
        "input": "integer",  # bool likely accepted too, but using integer
        "other": "integer"   # bool likely accepted too, but using integer
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.block_diag"] = {
    "args": {
        "tensors": "tensor_list"  # corresponds to *tensors
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.broadcast_tensors"] = {
    "args": {
        "tensors": "tensor_list"  # varargs accepted; represent as a list of tensors
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.broadcast_to_1"] = {
    "args": {
        "input": "tensor",
        "size": "tuple"  # torch.Size acceptable; treated as tuple
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.broadcast_to_2"] = {
    "args": {
        "input": "tensor",
        "size": "list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.cartesian_prod"] = {
    "args": {
        "tensors": "tensor_list"  # varargs of 1D tensors
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.chain_matmul"] = {
    "args": {
        "matrices": "tensor_list"  # actually varargs: *matrices
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.channel_shuffle"] = {
    "args": {
        "input": "tensor",
        "groups": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.cholesky_solve"] = {
    "args": {
        "input": "tensor",
        "input2": "tensor"
    },
    "kwargs": {
        "upper": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.cholesky"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "upper": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.chunk"] = {
    "args": {
        "input": "tensor",
        "chunks": "integer"
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.clip_1"] = {
    "args": {
        "input": "tensor",
        "min": "float",  # number (int or float)
        "max": "float",  # number (int or float)
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.clip_2"] = {
    "args": {
        "input": "tensor",
        "min": "tensor",
        "max": "tensor",
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.clip_3"] = {
    "args": {
        "input": "tensor",
        "min": "float",  # number (int or float)
        "max": "tensor",
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.clip_4"] = {
    "args": {
        "input": "tensor",
        "min": "tensor",
        "max": "float",  # number (int or float)
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.combinations"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "r": "integer",
        "with_replacement": "boolean"
    },
    "inner": {}
}
signatures["torch.complex"] = {
    "args": {
        "real": "tensor",
        "imag": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.copysign_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}

signatures["torch.copysign_2"] = {
    "args": {
        "input": "tensor",
        "other": "float"  # Number could be float or int; separate int variant below
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}

signatures["torch.copysign_3"] = {
    "args": {
        "input": "tensor",
        "other": "integer"  # Number could be int or float; float variant above
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.count_nonzero_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.count_nonzero_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "tuple"  # tuple of integers
    },
    "inner": {},
}
signatures["torch.cross"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "dim": "integer",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.cumprod"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "dtype": "dtype",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.deg2rad"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.det"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # Alias of torch.linalg.det, which supports out
    },
    "inner": {},
}
signatures["torch.diag_embed"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "offset": "integer",
        "dim1": "integer",
        "dim2": "integer",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.diagflat"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "offset": "integer"
    },
    "inner": {},
}
signatures["torch.diagonal"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "offset": "integer",
        "dim1": "integer",
        "dim2": "integer"
    },
    "inner": {},
}
signatures["torch.diag"] = {
    "args": {
        "input": "tensor",
        "diagonal": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.dist"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "p": "float"  # could also accept integers, but treating as float per docs
    },
    "inner": {},
}
signatures["torch.dot"] = {
    "args": {
        "input": "tensor",
        "tensor": "tensor"  # a.k.a. 'other' in PyTorch
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.dsplit_1"] = {
    "args": {
        "input": "tensor",
        "indices_or_sections": "integer"
    },
    "kwargs": {},
    "inner": {},
}

signatures["torch.dsplit_2"] = {
    "args": {
        "input": "tensor",
        "indices_or_sections": "list"  # list of ints
    },
    "kwargs": {},
    "inner": {},
}

signatures["torch.dsplit_3"] = {
    "args": {
        "input": "tensor",
        "indices_or_sections": "tuple"  # tuple of ints
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.fix"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.flatten"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "start_dim": "integer",
        "end_dim": "integer"
    },
    "inner": {},
}
signatures["torch.flip_1"] = {
    "args": {
        "input": "tensor",
        "dims": "list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.flip_2"] = {
    "args": {
        "input": "tensor",
        "dims": "tuple"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.flipud"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.float_power_1"] = {
    "args": {
        "input": "tensor",
        "exponent": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.float_power_2"] = {
    "args": {
        "input": "tensor",
        "exponent": "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.float_power_3"] = {
    "args": {
        "input": "tensor",
        "exponent": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.float_power_4"] = {
    "args": {
        "input": "float",
        "exponent": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.float_power_5"] = {
    "args": {
        "input": "integer",
        "exponent": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.floor_divide_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.floor_divide_2"] = {
    "args": {
        "input": "tensor",
        "other": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.floor_divide_3"] = {
    "args": {
        "input": "tensor",
        "other": "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.full_like_1"] = {
    "args": {
        "input": "tensor",
        "fill_value": "integer"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string",  # should be layout enum
        "requires_grad": "boolean",
        "memory_format": "string",  # should be memory_format enum
        "pin_memory": "boolean"
    },
    "inner": {},
}
signatures["torch.full_like_2"] = {
    "args": {
        "input": "tensor",
        "fill_value": "float"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string",  # should be layout enum
        "requires_grad": "boolean",
        "memory_format": "string",  # should be memory_format enum
        "pin_memory": "boolean"
    },
    "inner": {},
}
signatures["torch.full_like_3"] = {
    "args": {
        "input": "tensor",
        "fill_value": "boolean"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string",  # should be layout enum
        "requires_grad": "boolean",
        "memory_format": "string",  # should be memory_format enum
        "pin_memory": "boolean"
    },
    "inner": {},
}
signatures["torch.gather"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "tensor"  # LongTensor in docs, mapped to "tensor"
    },
    "kwargs": {
        "sparse_grad": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.gradient_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "spacing": "float",
        "dim": "integer",
        "edge_order": "integer"
    },
    "inner": {},
}

signatures["torch.gradient_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "spacing": "float",
        "dim": "list",
        "edge_order": "integer"
    },
    "inner": {},
}

signatures["torch.gradient_3"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "spacing": "list",  # list of scalars; tuples also work in practice
        "dim": "integer",
        "edge_order": "integer"
    },
    "inner": {},
}

signatures["torch.gradient_4"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "spacing": "list",  # list of scalars; tuples also work in practice
        "dim": "list",
        "edge_order": "integer"
    },
    "inner": {},
}

signatures["torch.gradient_5"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "spacing": "tensor_list",  # list of 1D tensors; tuples also work in practice
        "dim": "integer",
        "edge_order": "integer"
    },
    "inner": {},
}

signatures["torch.gradient_6"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "spacing": "tensor_list",  # list of 1D tensors; tuples also work in practice
        "dim": "list",
        "edge_order": "integer"
    },
    "inner": {},
}
signatures["torch.greater_equal_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.greater_equal_2"] = {
    "args": {
        "input": "tensor",
        "other": "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.greater_equal_3"] = {
    "args": {
        "input": "tensor",
        "other": "integer"  # bool is also accepted; using integer as closest allowed type
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.greater_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.greater_2"] = {
    "args": {
        "input": "tensor",
        "other": "integer"  # Number allowed; using integer variant
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.greater_3"] = {
    "args": {
        "input": "tensor",
        "other": "float"  # Number allowed; using float variant
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.greater_4"] = {
    "args": {
        "input": "tensor",
        "other": "boolean"  # Number allowed; bool also accepted in PyTorch comparisons
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.hsplit_1"] = {
    "args": {
        "input": "tensor",
        "indices_or_sections": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.hsplit_2"] = {
    "args": {
        "input": "tensor",
        "indices_or_sections": "list"  # list of ints
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.hsplit_3"] = {
    "args": {
        "input": "tensor",
        "indices_or_sections": "tuple"  # tuple of ints
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.imag"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.index_add"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "tensor",
        "source": "tensor"
    },
    "kwargs": {
        "alpha": "float"  # could be any numeric scalar
    },
    "inner": {},
}
signatures["torch.index_copy"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "tensor",
        "source": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # may not be available in all versions
    },
    "inner": {},
}
signatures["torch.index_select"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "tensor"  # should be a Long/Int tensor of indices
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.inner"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.inverse"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.isclose"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "rtol": "float",
        "atol": "float",
        "equal_nan": "boolean"
    },
    "inner": {},
}
signatures["torch.isinf"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        # Some PyTorch versions have an 'out' tensor kwarg, but it's not in the provided docs.
    },
    "inner": {},
}
signatures["torch.isin_1"] = {
    "args": {
        "elements": "tensor",
        "test_elements": "tensor"
    },
    "kwargs": {
        "assume_unique": "boolean",
        "invert": "boolean"
    },
    "inner": {},
}

signatures["torch.isin_2"] = {
    "args": {
        "elements": "tensor",
        "test_elements": "integer"  # scalar; could also be a zero-dim tensor, but treated as Python int here
    },
    "kwargs": {
        "assume_unique": "boolean",
        "invert": "boolean"
    },
    "inner": {},
}

signatures["torch.isin_3"] = {
    "args": {
        "elements": "tensor",
        "test_elements": "float"  # scalar; could also be a zero-dim tensor, but treated as Python float here
    },
    "kwargs": {
        "assume_unique": "boolean",
        "invert": "boolean"
    },
    "inner": {},
}

signatures["torch.isin_4"] = {
    "args": {
        "elements": "tensor",
        "test_elements": "boolean"  # scalar; could also be a zero-dim tensor, but treated as Python bool here
    },
    "kwargs": {
        "assume_unique": "boolean",
        "invert": "boolean"
    },
    "inner": {},
}

signatures["torch.isin_5"] = {
    "args": {
        "elements": "integer",  # scalar; could also be a zero-dim tensor, but treated as Python int here
        "test_elements": "tensor"
    },
    "kwargs": {
        "assume_unique": "boolean",
        "invert": "boolean"
    },
    "inner": {},
}

signatures["torch.isin_6"] = {
    "args": {
        "elements": "float",  # scalar; could also be a zero-dim tensor, but treated as Python float here
        "test_elements": "tensor"
    },
    "kwargs": {
        "assume_unique": "boolean",
        "invert": "boolean"
    },
    "inner": {},
}

signatures["torch.isin_7"] = {
    "args": {
        "elements": "boolean",  # scalar; could also be a zero-dim tensor, but treated as Python bool here
        "test_elements": "tensor"
    },
    "kwargs": {
        "assume_unique": "boolean",
        "invert": "boolean"
    },
    "inner": {},
}
signatures["torch.isnan"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_nonzero"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.isreal"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.kron"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.ldexp"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.less_equal_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # Bool tensor in practice
    },
    "inner": {},
}
signatures["torch.less_equal_2"] = {
    "args": {
        "input": "tensor",
        "other": "float"
    },
    "kwargs": {
        "out": "tensor"  # Bool tensor in practice
    },
    "inner": {},
}
signatures["torch.less_equal_3"] = {
    "args": {
        "input": "tensor",
        "other": "integer"
    },
    "kwargs": {
        "out": "tensor"  # Bool tensor in practice
    },
    "inner": {},
}
signatures["torch.less_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.less_2"] = {
    "args": {
        "input": "tensor",
        "other": "float"  # could also be integer; see _3
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.less_3"] = {
    "args": {
        "input": "tensor",
        "other": "integer"  # could also be float; see _2
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.logcumsumexp"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.logdet"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.logical_and"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # expects bool tensor if inputs are bool
    },
    "inner": {}
}
signatures["torch.logical_not"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # optional output tensor
    },
    "inner": {},
}
signatures["torch.logical_or_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.logical_or_2"] = {
    "args": {
        "input": "tensor",
        "other": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.logical_or_3"] = {
    "args": {
        "input": "tensor",
        "other": "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.logical_or_4"] = {
    "args": {
        "input": "tensor",
        "other": "boolean"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.logical_xor"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.logit"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "eps": "float",  # can be None
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.logsumexp_1"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.logsumexp_2"] = {
    "args": {
        "input": "tensor",
        "dim": "tuple"  # tuple of integers
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.lu_solve"] = {
    "args": {
        "B": "tensor",
        "LU_data": "tensor",
        "LU_pivots": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.masked_scatter"] = {
    "args": {
        "input": "tensor",
        "mask": "tensor",
        "source": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.matmul"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # Note: not supported for 1D dot-product case per docs
    },
    "inner": {},
}
signatures["torch.matrix_exp"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.matrix_power"] = {
    "args": {
        "input": "tensor",
        "n": "integer"
    },
    "kwargs": {
        "out": "tensor"  # optional; some versions may omit this
    },
    "inner": {},
}
signatures["torch.max_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}

signatures["torch.max_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tuple"  # tuple of (Tensor, Tensor)
    },
    "inner": {},
}

signatures["torch.max_3"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.median_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.median_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"  # optional; None is also allowed in docs
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tuple"  # (Tensor, Tensor)
    },
    "inner": {},
}
signatures["torch.min_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}

signatures["torch.min_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tuple"  # tuple of (min, min_indices) tensors
    },
    "inner": {},
}

signatures["torch.min_3"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.moveaxis_1"] = {
    "args": {
        "input": "tensor",
        "source": "integer",
        "destination": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.moveaxis_2"] = {
    "args": {
        "input": "tensor",
        "source": "tuple",
        "destination": "tuple"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.moveaxis_3"] = {
    "args": {
        "input": "tensor",
        "source": "list",
        "destination": "list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.movedim_1"] = {
    "args": {
        "input": "tensor",
        "source": "integer",
        "destination": "integer"
    },
    "kwargs": {},
    "inner": {},
}

signatures["torch.movedim_2"] = {
    "args": {
        "input": "tensor",
        "source": "tuple",  # could also accept list[int] in practice
        "destination": "tuple"  # could also accept list[int] in practice
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.msort"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.mul_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}

# Note: PyTorch also accepts complex numbers for 'other'; not representable in allowed types here.
signatures["torch.mul_2"] = {
    "args": {
        "input": "tensor",
        "other": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}

signatures["torch.mul_3"] = {
    "args": {
        "input": "tensor",
        "other": "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.mv"] = {
    "args": {
        "input": "tensor",
        "vec": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nanmedian_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nanmedian_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer",  # None is also allowed
        "keepdim": "boolean",
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.nan_to_num"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "nan": "float",       # can be None; using float
        "posinf": "float",    # can be None; using float
        "neginf": "float",    # can be None; using float
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.narrow_1"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "start": "integer",
        "length": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.narrow_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "start": "tensor",  # 0-dim integral tensor
        "length": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.negative"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.not_equal_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.not_equal_2"] = {
    "args": {
        "input": "tensor",
        "other": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.not_equal_3"] = {
    "args": {
        "input": "tensor",
        "other": "float"  # could also accept complex numbers, mapped to float here
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.not_equal_4"] = {
    "args": {
        "input": "tensor",
        "other": "boolean"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.ones_like"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string",  # should be layout enum
        "requires_grad": "boolean",
        "memory_format": "string"  # should be memory_format enum
    },
    "inner": {},
}
signatures["torch.outer"] = {
    "args": {
        "input": "tensor",
        "vec2": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.pdist"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "p": "float"
    },
    "inner": {},
}
signatures["torch.permute"] = {
    "args": {
        "input": "tensor",
        "dims": "tuple"  # tuple of int; in practice may accept list of int
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.pinverse"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "rcond": "float"  # could also be a tensor in torch.linalg.pinv, but torch.pinverse docs show float
    },
    "inner": {},
}
signatures["torch.polar"] = {
    "args": {
        "abs": "tensor",
        "angle": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # complex tensor matching float/double inputs
    },
    "inner": {},
}
signatures["torch.prod_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.prod_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"  # could be None in docs, but we only allow listed types
    },
    "kwargs": {
        "keepdim": "boolean",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.put"] = {
    "args": {
        "input": "tensor",
        "index": "tensor",
        "source": "tensor"
    },
    "kwargs": {
        "accumulate": "boolean"
    },
    "inner": {},
}
signatures["torch.rad2deg"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # optional
    },
    "inner": {},
}
signatures["torch.randint_like"] = {
    "args": {
        "input": "tensor",
        "high": "integer"
    },
    "kwargs": {
        "low": "integer",
        "dtype": "dtype",
        "layout": "string",  # should be torch.layout enum
        "pin_memory": "boolean",
        "requires_grad": "boolean",
        "memory_format": "string",  # should be torch.memory_format enum
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.remainder_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.remainder_2"] = {
    "args": {
        "input": "tensor",
        "other": "integer"  # scalar number
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.remainder_3"] = {
    "args": {
        "input": "tensor",
        "other": "float"  # scalar number
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.remainder_4"] = {
    "args": {
        "input": "integer",  # scalar number
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.remainder_5"] = {
    "args": {
        "input": "float",  # scalar number
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.remainder_6"] = {
    "args": {
        "input": "integer",  # scalar number
        "other": "integer"   # scalar number
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.remainder_7"] = {
    "args": {
        "input": "float",  # scalar number
        "other": "float"   # scalar number
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.remainder_8"] = {
    "args": {
        "input": "integer",  # scalar number
        "other": "float"     # scalar number
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.remainder_9"] = {
    "args": {
        "input": "float",    # scalar number
        "other": "integer"   # scalar number
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.rot90_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "k": "integer",
        "dims": "list"  # list of two integers
    },
    "inner": {},
}
signatures["torch.rot90_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "k": "integer",
        "dims": "tuple"  # tuple of two integers
    },
    "inner": {},
}
signatures["torch.scatter_1"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "tensor",
        "src": "tensor"
    },
    "kwargs": {
        "reduce": "string"
    },
    "inner": {},
}
signatures["torch.scatter_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "tensor",
        "value": "float"  # could also be integer
    },
    "kwargs": {
        "reduce": "string"
    },
    "inner": {},
}
signatures["torch.slogdet"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tuple"  # tuple of two tensors (sign, logabsdet)
    },
    "inner": {},
}
signatures["torch.square"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.std_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer",
        "correction": "integer",
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.std_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "tuple",  # tuple of ints
        "correction": "integer",
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.std_3"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "list",  # list of ints
        "correction": "integer",
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.subtract_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "alpha": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.subtract_2"] = {
    "args": {
        "input": "tensor",
        "other": "float"  # could also be integer; using float for general Number
    },
    "kwargs": {
        "alpha": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.subtract_3"] = {
    "args": {
        "input": "float",  # could also be integer; using float for general Number
        "other": "tensor"
    },
    "kwargs": {
        "alpha": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.sum_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {},
}

signatures["torch.sum_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "dtype": "dtype"
    },
    "inner": {},
}

signatures["torch.sum_3"] = {
    "args": {
        "input": "tensor",
        "dim": "tuple"
    },
    "kwargs": {
        "keepdim": "boolean",
        "dtype": "dtype"
    },
    "inner": {},
}

signatures["torch.sum_4"] = {
    "args": {
        "input": "tensor",
        "dim": "list"  # list of integers
    },
    "kwargs": {
        "keepdim": "boolean",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.swapaxes"] = {
    "args": {
        "input": "tensor",
        "axis0": "integer",
        "axis1": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.swapdims"] = {
    "args": {
        "input": "tensor",
        "dim0": "integer",
        "dim1": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.tile_1"] = {
    "args": {
        "input": "tensor",
        "dims": "tuple"  # tuple of integers
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.tile_2"] = {
    "args": {
        "input": "tensor",
        "dims": "list"  # list of integers
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.transpose"] = {
    "args": {
        "input": "tensor",
        "dim0": "integer",
        "dim1": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.trapz_1"] = {
    "args": {
        "y": "tensor"
    },
    "kwargs": {
        "dx": "float",
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.trapz_2"] = {
    "args": {
        "y": "tensor",
        "x": "tensor"  # x is sample points (1-D tensor) along the integration dim
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.t"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.unbind"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.unsafe_split_with_sizes_1"] = {
    "args": {
        "input": "tensor",
        "split_sizes": "list"
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.unsafe_split_with_sizes_2"] = {
    "args": {
        "input": "tensor",
        "split_sizes": "tuple"
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.unsqueeze"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.vander"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "N": "integer",
        "increasing": "boolean"
    },
    "inner": {},
}
signatures["torch.vdot"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.vsplit_1"] = {
    "args": {
        "input": "tensor",
        "indices_or_sections": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.vsplit_2"] = {
    "args": {
        "input": "tensor",
        "indices_or_sections": "list"  # list of ints
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.vsplit_3"] = {
    "args": {
        "input": "tensor",
        "indices_or_sections": "tuple"  # tuple of ints
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.zeros_like"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string",  # should be torch.layout
        "requires_grad": "boolean",
        "memory_format": "string"  # should be torch.memory_format
    },
    "inner": {},
}
signatures["torch.empty_1"] = {
    "args": {
        "size": "tuple"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string",  # should be torch.layout
        "requires_grad": "boolean",
        "pin_memory": "boolean",
        "memory_format": "string"  # should be torch.memory_format
    },
    "inner": {},
}
signatures["torch.empty_2"] = {
    "args": {
        "size": "list"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string",  # should be torch.layout
        "requires_grad": "boolean",
        "pin_memory": "boolean",
        "memory_format": "string"  # should be torch.memory_format
    },
    "inner": {},
}
signatures["torch.empty_3"] = {
    "args": {
        "size": "integer"  # represents 1-D case; varargs of integers are allowed in real API
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string",  # should be torch.layout
        "requires_grad": "boolean",
        "pin_memory": "boolean",
        "memory_format": "string"  # should be torch.memory_format
    },
    "inner": {},
}
signatures["torch.empty_like"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string",  # should be torch.layout
        "requires_grad": "boolean",
        "pin_memory": "boolean",
        "memory_format": "string"  # should be torch.memory_format
    },
    "inner": {},
}
signatures["torch.empty_strided_1"] = {
    "args": {
        "size": "tuple",
        "stride": "tuple"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string",  # actual type is torch.layout
        "requires_grad": "boolean",
        "pin_memory": "boolean"
    },
    "inner": {},
}
signatures["torch.empty_strided_2"] = {
    "args": {
        "size": "list",
        "stride": "list"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string",  # actual type is torch.layout
        "requires_grad": "boolean",
        "pin_memory": "boolean"
    },
    "inner": {},
}
signatures["torch.empty_strided_3"] = {
    "args": {
        "size": "tuple",
        "stride": "list"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string",  # actual type is torch.layout
        "requires_grad": "boolean",
        "pin_memory": "boolean"
    },
    "inner": {},
}
signatures["torch.empty_strided_4"] = {
    "args": {
        "size": "list",
        "stride": "tuple"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string",  # actual type is torch.layout
        "requires_grad": "boolean",
        "pin_memory": "boolean"
    },
    "inner": {},
}
signatures["torch.fft.fftshift_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.fft.fftshift_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.fft.fftshift_3"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "tuple"  # Tuple[int, ...]
    },
    "inner": {},
}
signatures["torch.fft.ifftshift_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.fft.ifftshift_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "tuple"
    },
    "inner": {},
}
signatures["torch.linalg.eigvals"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.linalg.householder_product"] = {
    "args": {
        "A": "tensor",
        "tau": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.linalg.inv"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.linalg.matrix_power"] = {
    "args": {
        "A": "tensor",
        "n": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.linalg.pinv_1"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "atol": "float",
        "rtol": "float",
        "rcond": "float",  # alias for rtol
        "hermitian": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.linalg.pinv_2"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "atol": "tensor",
        "rtol": "tensor",
        "rcond": "tensor",  # alias for rtol
        "hermitian": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.linalg.solve"] = {
    "args": {
        "A": "tensor",
        "B": "tensor"
    },
    "kwargs": {
        "left": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.linalg.svdvals"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "driver": "string",  # can be None; string options: "gesvd", "gesvdj", "gesvda"
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.linalg.tensorinv"] = {
    "args": {
        "A": "tensor",
        "ind": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.AdaptiveAvgPool1d_1"] = {
    "args": {
        "output_size": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AdaptiveAvgPool1d_2"] = {
    "args": {
        "output_size": "tuple"  # expected to be a 1-element tuple of int
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AdaptiveAvgPool2d_1"] = {
    "args": {
        "output_size": "integer"  # can also be None (not representable here)
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AdaptiveAvgPool2d_2"] = {
    "args": {
        "output_size": "tuple"  # elements can be integer or None
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AdaptiveMaxPool1d_1"] = {
    "args": {
        "output_size": "integer"
    },
    "kwargs": {
        "return_indices": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AdaptiveMaxPool1d_2"] = {
    "args": {
        "output_size": "tuple"  # tuple[int]
    },
    "kwargs": {
        "return_indices": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AdaptiveMaxPool2d_1"] = {
    "args": {
        "output_size": "integer"  # could also be None (not representable here)
    },
    "kwargs": {
        "return_indices": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AdaptiveMaxPool2d_2"] = {
    "args": {
        "output_size": "tuple"  # elements can be int or None
    },
    "kwargs": {
        "return_indices": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AdaptiveMaxPool3d_1"] = {
    "args": {
        "output_size": "integer"
    },
    "kwargs": {
        "return_indices": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}

signatures["torch.nn.AdaptiveMaxPool3d_2"] = {
    "args": {
        "output_size": "tuple"  # elements can be int or None; entire arg may also be None in PyTorch
    },
    "kwargs": {
        "return_indices": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AlphaDropout"] = {
    "args": {},
    "kwargs": {
        "p": "float",
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AvgPool1d_1"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AvgPool1d_2"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AvgPool1d_3"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AvgPool1d_4"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AvgPool1d_5"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AvgPool1d_6"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AvgPool1d_7"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AvgPool1d_8"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AvgPool2d_1"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"  # Optional[int]
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AvgPool2d_2"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"  # Optional[int]
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AvgPool2d_3"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"  # Optional[int]
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AvgPool2d_4"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"  # Optional[int]
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AvgPool2d_5"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"  # Optional[int]
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AvgPool2d_6"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"  # Optional[int]
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AvgPool2d_7"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"  # Optional[int]
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AvgPool2d_8"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"  # Optional[int]
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AvgPool3d_1"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"  # optional int
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AvgPool3d_2"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"  # optional int
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AvgPool3d_3"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"  # optional int
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AvgPool3d_4"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"  # optional int
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AvgPool3d_5"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"  # optional int
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AvgPool3d_6"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"  # optional int
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AvgPool3d_7"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"  # optional int
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AvgPool3d_8"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"  # optional int
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.BCEWithLogitsLoss"] = {
    "args": {},
    "kwargs": {
        "weight": "tensor",
        "size_average": "boolean",  # can be None
        "reduce": "boolean",        # can be None
        "reduction": "string",
        "pos_weight": "tensor"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.BatchNorm1d"] = {
    "args": {
        "num_features": "integer"
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",  # Optional[float]; can be None
        "affine": "boolean",
        "track_running_stats": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.BatchNorm2d"] = {
    "args": {
        "num_features": "integer"
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",  # can be None (cumulative moving average)
        "affine": "boolean",
        "track_running_stats": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.BatchNorm3d"] = {
    "args": {
        "num_features": "integer"
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",  # Can be None for cumulative moving average
        "affine": "boolean",
        "track_running_stats": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.CELU"] = {
    "args": {},
    "kwargs": {
        "alpha": "float",
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ConstantPad1d_1"] = {
    "args": {
        "padding": "integer",
        "value": "float"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ConstantPad1d_2"] = {
    "args": {
        "padding": "tuple",
        "value": "float"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ConstantPad2d_1"] = {
    "args": {
        "padding": "integer",
        "value": "float"  # could also be integer in practice
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ConstantPad2d_2"] = {
    "args": {
        "padding": "tuple",  # expected length 4: (left, right, top, bottom)
        "value": "float"  # could also be integer in practice
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ConstantPad3d_1"] = {
    "args": {
        "padding": "integer",
        "value": "float"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ConstantPad3d_2"] = {
    "args": {
        "padding": "tuple",
        "value": "float"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.Dropout"] = {
    "args": {},
    "kwargs": {
        "p": "float",
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.CosineEmbeddingLoss"] = {
    "args": {},
    "kwargs": {
        "margin": "float",
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {
        "args": {
            "input1": "tensor",
            "input2": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.CosineSimilarity"] = {
    "args": {},
    "kwargs": {
        "dim": "integer",
        "eps": "float"
    },
    "inner": {
        "args": {
            "x1": "tensor",
            "x2": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.Dropout2d"] = {
    "args": {},
    "kwargs": {
        "p": "float",
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.Dropout3d"] = {
    "args": {},
    "kwargs": {
        "p": "float",
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ELU"] = {
    "args": {},
    "kwargs": {
        "alpha": "float",
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.FeatureAlphaDropout"] = {
    "args": {},
    "kwargs": {
        "p": "float",
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.Flatten"] = {
    "args": {},
    "kwargs": {
        "start_dim": "integer",
        "end_dim": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.Fold_1"] = {
    "args": {
        "output_size": "integer",
        "kernel_size": "integer"
    },
    "kwargs": {
        "dilation": "integer",
        "padding": "integer",
        "stride": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}

signatures["torch.nn.Fold_2"] = {
    "args": {
        "output_size": "tuple",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "dilation": "tuple",
        "padding": "tuple",
        "stride": "tuple"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.GELU"] = {
    "args": {},
    "kwargs": {
        "approximate": "string"  # expects 'none' or 'tanh'
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.GLU"] = {
    "args": {},
    "kwargs": {
        "dim": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.GroupNorm"] = {
    "args": {
        "num_groups": "integer",
        "num_channels": "integer"
    },
    "kwargs": {
        "eps": "float",
        "affine": "boolean",
        "dtype": "dtype"  # could be None or torch.dtype; using dtype
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.Hardshrink"] = {
    "args": {},
    "kwargs": {
        "lambd": "float"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.Hardtanh"] = {
    "args": {},
    "kwargs": {
        "min_val": "float",
        "max_val": "float",
        "inplace": "boolean",
        "min_value": "float",  # deprecated alias of min_val
        "max_value": "float"   # deprecated alias of max_val
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.HingeEmbeddingLoss"] = {
    "args": {},
    "kwargs": {
        "margin": "float",
        "size_average": "boolean",  # technically Optional[bool]
        "reduce": "boolean",        # technically Optional[bool]
        "reduction": "string"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.HuberLoss"] = {
    "args": {},
    "kwargs": {
        "reduction": "string",
        "delta": "float"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.Identity"] = {
    "args": {},
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.InstanceNorm1d"] = {
    "args": {
        "num_features": "integer"
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean",
        "dtype": "dtype"  # optional
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.InstanceNorm2d"] = {
    "args": {
        "num_features": "integer"
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",  # Optional[float] in docs
        "affine": "boolean",
        "track_running_stats": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.InstanceNorm3d"] = {
    "args": {
        "num_features": "integer"
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",  # Optional[float]; None is also accepted by PyTorch
        "affine": "boolean",
        "track_running_stats": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.KLDivLoss"] = {
    "args": {},
    "kwargs": {
        "size_average": "boolean",  # can be None
        "reduce": "boolean",        # can be None
        "reduction": "string",
        "log_target": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.L1Loss"] = {
    "args": {},
    "kwargs": {
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.LPPool1d_1"] = {
    "args": {
        "norm_type": "float",  # could also accept integer
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}

signatures["torch.nn.LPPool1d_2"] = {
    "args": {
        "norm_type": "float",  # could also accept integer
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}

signatures["torch.nn.LPPool1d_3"] = {
    "args": {
        "norm_type": "float",  # could also accept integer
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}

signatures["torch.nn.LPPool1d_4"] = {
    "args": {
        "norm_type": "float",  # could also accept integer
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.LPPool2d_1"] = {
    "args": {
        "norm_type": "float",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.LPPool2d_2"] = {
    "args": {
        "norm_type": "float",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.LPPool2d_3"] = {
    "args": {
        "norm_type": "float",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.LPPool2d_4"] = {
    "args": {
        "norm_type": "float",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.LayerNorm_1"] = {
    "args": {
        "normalized_shape": "integer"
    },
    "kwargs": {
        "eps": "float",
        "elementwise_affine": "boolean",
        "bias": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.LayerNorm_2"] = {
    "args": {
        "normalized_shape": "list"
    },
    "kwargs": {
        "eps": "float",
        "elementwise_affine": "boolean",
        "bias": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.LayerNorm_3"] = {
    "args": {
        "normalized_shape": "tuple"  # torch.Size treated as tuple
    },
    "kwargs": {
        "eps": "float",
        "elementwise_affine": "boolean",
        "bias": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.LeakyReLU"] = {
    "args": {},
    "kwargs": {
        "negative_slope": "float",
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.LocalResponseNorm"] = {
    "args": {
        "size": "integer"
    },
    "kwargs": {
        "alpha": "float",
        "beta": "float",
        "k": "float"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.LogSigmoid"] = {
    "args": {},
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.LogSoftmax_1"] = {
    "args": {
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}

signatures["torch.nn.LogSoftmax_2"] = {
    "args": {},
    "kwargs": {
        "dim": "integer"  # optional in constructor; defaults to None in docs
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.MSELoss"] = {
    "args": {},
    "kwargs": {
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.MarginRankingLoss"] = {
    "args": {},
    "kwargs": {
        "margin": "float",
        "size_average": "boolean",  # can be None in PyTorch, using boolean here
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {
        "args": {
            "input1": "tensor",
            "input2": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.MaxPool2d_1"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",  # or tuple
        "padding": "integer",  # or tuple
        "dilation": "integer",  # or tuple
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.MaxPool2d_2"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",  # or tuple
        "padding": "integer",  # or tuple
        "dilation": "integer",  # or tuple
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.Mish"] = {
    "args": {},
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.MultiLabelMarginLoss"] = {
    "args": {},
    "kwargs": {
        "size_average": "boolean",  # can be None in PyTorch, but using boolean per spec
        "reduce": "boolean",        # can be None in PyTorch, but using boolean per spec
        "reduction": "string"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.MultiLabelSoftMarginLoss"] = {
    "args": {},
    "kwargs": {
        "weight": "tensor",
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.MultiMarginLoss"] = {
    "args": {},
    "kwargs": {
        "p": "integer",
        "margin": "float",
        "weight": "tensor",  # optional, or None
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.PairwiseDistance"] = {
    "args": {},
    "kwargs": {
        "p": "float",
        "eps": "float",
        "keepdim": "boolean"
    },
    "inner": {
        "args": {
            "x1": "tensor",
            "x2": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.PixelShuffle"] = {
    "args": {
        "upscale_factor": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.PixelUnshuffle"] = {
    "args": {
        "downscale_factor": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.PoissonNLLLoss"] = {
    "args": {},
    "kwargs": {
        "log_input": "boolean",
        "full": "boolean",
        "size_average": "boolean",
        "eps": "float",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.RReLU"] = {
    "args": {},
    "kwargs": {
        "lower": "float",
        "upper": "float",
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ReLU"] = {
    "args": {},
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ReLU6"] = {
    "args": {},
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ReflectionPad1d_1"] = {
    "args": {
        "padding": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ReflectionPad1d_2"] = {
    "args": {
        "padding": "tuple"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ReflectionPad2d_1"] = {
    "args": {
        "padding": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ReflectionPad2d_2"] = {
    "args": {
        "padding": "tuple"  # tuple of 4 ints: (left, right, top, bottom)
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ReflectionPad3d_1"] = {
    "args": {
        "padding": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ReflectionPad3d_2"] = {
    "args": {
        "padding": "tuple"  # expected to be a 6-tuple (left, right, top, bottom, front, back)
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ReplicationPad1d_1"] = {
    "args": {
        "padding": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ReplicationPad1d_2"] = {
    "args": {
        "padding": "tuple"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ReplicationPad2d_1"] = {
    "args": {
        "padding": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}

signatures["torch.nn.ReplicationPad2d_2"] = {
    "args": {
        "padding": "tuple"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ReplicationPad3d_1"] = {
    "args": {
        "padding": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ReplicationPad3d_2"] = {
    "args": {
        "padding": "tuple"  # expected to be a 6-tuple
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.SELU"] = {
    "args": {},
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.SiLU"] = {
    "args": {},
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.Sigmoid"] = {
    "args": {},
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.SmoothL1Loss"] = {
    "args": {},
    "kwargs": {
        "size_average": "boolean",  # deprecated
        "reduce": "boolean",        # deprecated
        "reduction": "string",
        "beta": "float"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.SoftMarginLoss"] = {
    "args": {},
    "kwargs": {
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.Softmax"] = {
    "args": {},
    "kwargs": {
        "dim": "integer"  # can be None as default
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.Softmax2d"] = {
    "args": {},
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.Softmin"] = {
    "args": {},
    "kwargs": {
        "dim": "integer"  # can be omitted (defaults to None in PyTorch)
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.Softplus"] = {
    "args": {},
    "kwargs": {
        "beta": "float",
        "threshold": "float"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.Softshrink"] = {
    "args": {},
    "kwargs": {
        "lambd": "float"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.Softsign"] = {
    "args": {},
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.Tanh"] = {
    "args": {},
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.Tanhshrink"] = {
    "args": {},
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.Threshold"] = {
    "args": {
        "threshold": "float",
        "value": "float"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.TripletMarginLoss"] = {
    "args": {},
    "kwargs": {
        "margin": "float",
        "p": "integer",
        "eps": "float",
        "swap": "boolean",
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {
        "args": {
            "anchor": "tensor",
            "positive": "tensor",
            "negative": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ZeroPad2d_1"] = {
    "args": {
        "padding": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ZeroPad2d_2"] = {
    "args": {
        "padding": "tuple"  # expected length 4: (left, right, top, bottom)
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.functional.adaptive_avg_pool1d_1"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer"
    },
    "kwargs": {},
    "inner": {},
}

signatures["torch.nn.functional.adaptive_avg_pool1d_2"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple"  # Likely a 1-element tuple[int]
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.adaptive_avg_pool2d_1"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.adaptive_avg_pool2d_2"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple"  # tuple of two integers (H, W)
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.adaptive_avg_pool3d_1"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.adaptive_avg_pool3d_2"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple"  # tuple of 3 integers
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.adaptive_max_pool1d_1"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer"  # can be a single target length
    },
    "kwargs": {
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.adaptive_max_pool1d_2"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple"  # likely a 1-element tuple for L_out
    },
    "kwargs": {
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.adaptive_max_pool2d_1"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer"  # single int applies to both H and W
    },
    "kwargs": {
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.adaptive_max_pool2d_2"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple"  # tuple of two ints (H, W)
    },
    "kwargs": {
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.adaptive_max_pool3d_1"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer"  # could also accept a sequence; using int variant here
    },
    "kwargs": {
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.adaptive_max_pool3d_2"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple"  # typically a tuple of three integers
    },
    "kwargs": {
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.affine_grid_1"] = {
    "args": {
        "theta": "tensor",
        "size": "tuple"
    },
    "kwargs": {
        "align_corners": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.affine_grid_2"] = {
    "args": {
        "theta": "tensor",
        "size": "list"  # could also be torch.Size (tuple-like)
    },
    "kwargs": {
        "align_corners": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.alpha_dropout"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "p": "float",
        "training": "boolean",
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.avg_pool3d_1"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"  # can be None
    },
    "inner": {},
}
signatures["torch.nn.functional.avg_pool3d_2"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"  # can be None
    },
    "inner": {},
}
signatures["torch.nn.functional.avg_pool3d_3"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"  # can be None
    },
    "inner": {},
}
signatures["torch.nn.functional.avg_pool3d_4"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"  # can be None
    },
    "inner": {},
}
signatures["torch.nn.functional.avg_pool3d_5"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"  # can be None
    },
    "inner": {},
}
signatures["torch.nn.functional.avg_pool3d_6"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"  # can be None
    },
    "inner": {},
}
signatures["torch.nn.functional.avg_pool3d_7"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"  # can be None
    },
    "inner": {},
}
signatures["torch.nn.functional.avg_pool3d_8"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"  # can be None
    },
    "inner": {},
}
signatures["torch.nn.functional.batch_norm"] = {
    "args": {
        "input": "tensor",
        "running_mean": "tensor",  # can be None
        "running_var": "tensor"    # can be None
    },
    "kwargs": {
        "weight": "tensor",  # optional; can be None
        "bias": "tensor",    # optional; can be None
        "training": "boolean",
        "momentum": "float",
        "eps": "float"
    },
    "inner": {},
}
signatures["torch.nn.functional.binary_cross_entropy"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "weight": "tensor",
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.nn.functional.binary_cross_entropy_with_logits"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "weight": "tensor",
        "size_average": "boolean",  # deprecated; can be None
        "reduce": "boolean",        # deprecated; can be None
        "reduction": "string",
        "pos_weight": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.functional.celu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "alpha": "float",
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.conv_transpose1d_1"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",  # can be None
        "stride": "integer",
        "padding": "integer",
        "output_padding": "integer",
        "groups": "integer",
        "dilation": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.conv_transpose1d_2"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",  # can be None
        "stride": "tuple",
        "padding": "tuple",
        "output_padding": "tuple",
        "groups": "integer",
        "dilation": "tuple"
    },
    "inner": {},
}
signatures["torch.nn.functional.conv_transpose2d_1"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",  # can be None
        "stride": "integer",
        "padding": "integer",
        "output_padding": "integer",
        "groups": "integer",
        "dilation": "integer",
        "padding_mode": "string"  # typically 'zeros'
    },
    "inner": {},
}
signatures["torch.nn.functional.conv_transpose2d_2"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",  # can be None
        "stride": "integer",
        "padding": "integer",
        "output_padding": "integer",
        "groups": "integer",
        "dilation": "tuple",
        "padding_mode": "string"  # typically 'zeros'
    },
    "inner": {},
}
signatures["torch.nn.functional.conv_transpose2d_3"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",  # can be None
        "stride": "integer",
        "padding": "integer",
        "output_padding": "tuple",
        "groups": "integer",
        "dilation": "integer",
        "padding_mode": "string"  # typically 'zeros'
    },
    "inner": {},
}
signatures["torch.nn.functional.conv_transpose2d_4"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",  # can be None
        "stride": "integer",
        "padding": "integer",
        "output_padding": "tuple",
        "groups": "integer",
        "dilation": "tuple",
        "padding_mode": "string"  # typically 'zeros'
    },
    "inner": {},
}
signatures["torch.nn.functional.conv_transpose2d_5"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",  # can be None
        "stride": "integer",
        "padding": "tuple",
        "output_padding": "integer",
        "groups": "integer",
        "dilation": "integer",
        "padding_mode": "string"  # typically 'zeros'
    },
    "inner": {},
}
signatures["torch.nn.functional.conv_transpose2d_6"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",  # can be None
        "stride": "integer",
        "padding": "tuple",
        "output_padding": "integer",
        "groups": "integer",
        "dilation": "tuple",
        "padding_mode": "string"  # typically 'zeros'
    },
    "inner": {},
}
signatures["torch.nn.functional.conv_transpose2d_7"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",  # can be None
        "stride": "integer",
        "padding": "tuple",
        "output_padding": "tuple",
        "groups": "integer",
        "dilation": "integer",
        "padding_mode": "string"  # typically 'zeros'
    },
    "inner": {},
}
signatures["torch.nn.functional.conv_transpose2d_8"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",  # can be None
        "stride": "integer",
        "padding": "tuple",
        "output_padding": "tuple",
        "groups": "integer",
        "dilation": "tuple",
        "padding_mode": "string"  # typically 'zeros'
    },
    "inner": {},
}
signatures["torch.nn.functional.conv_transpose2d_9"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",  # can be None
        "stride": "tuple",
        "padding": "integer",
        "output_padding": "integer",
        "groups": "integer",
        "dilation": "integer",
        "padding_mode": "string"  # typically 'zeros'
    },
    "inner": {},
}
signatures["torch.nn.functional.conv_transpose2d_10"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",  # can be None
        "stride": "tuple",
        "padding": "integer",
        "output_padding": "integer",
        "groups": "integer",
        "dilation": "tuple",
        "padding_mode": "string"  # typically 'zeros'
    },
    "inner": {},
}
signatures["torch.nn.functional.conv_transpose2d_11"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",  # can be None
        "stride": "tuple",
        "padding": "integer",
        "output_padding": "tuple",
        "groups": "integer",
        "dilation": "integer",
        "padding_mode": "string"  # typically 'zeros'
    },
    "inner": {},
}
signatures["torch.nn.functional.conv_transpose2d_12"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",  # can be None
        "stride": "tuple",
        "padding": "integer",
        "output_padding": "tuple",
        "groups": "integer",
        "dilation": "tuple",
        "padding_mode": "string"  # typically 'zeros'
    },
    "inner": {},
}
signatures["torch.nn.functional.conv_transpose2d_13"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",  # can be None
        "stride": "tuple",
        "padding": "tuple",
        "output_padding": "integer",
        "groups": "integer",
        "dilation": "integer",
        "padding_mode": "string"  # typically 'zeros'
    },
    "inner": {},
}
signatures["torch.nn.functional.conv_transpose2d_14"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",  # can be None
        "stride": "tuple",
        "padding": "tuple",
        "output_padding": "integer",
        "groups": "integer",
        "dilation": "tuple",
        "padding_mode": "string"  # typically 'zeros'
    },
    "inner": {},
}
signatures["torch.nn.functional.conv_transpose2d_15"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",  # can be None
        "stride": "tuple",
        "padding": "tuple",
        "output_padding": "tuple",
        "groups": "integer",
        "dilation": "integer",
        "padding_mode": "string"  # typically 'zeros'
    },
    "inner": {},
}
signatures["torch.nn.functional.conv_transpose2d_16"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",  # can be None
        "stride": "tuple",
        "padding": "tuple",
        "output_padding": "tuple",
        "groups": "integer",
        "dilation": "tuple",
        "padding_mode": "string"  # typically 'zeros'
    },
    "inner": {},
}
signatures["torch.nn.functional.cosine_embedding_loss"] = {
    "args": {
        "input1": "tensor",
        "input2": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "margin": "float",
        "size_average": "boolean",  # could be None; using boolean per allowed types
        "reduce": "boolean",        # could be None; using boolean per allowed types
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.nn.functional.cosine_similarity"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {
        "dim": "integer",
        "eps": "float"
    },
    "inner": {},
}
signatures["torch.nn.functional.cross_entropy"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "weight": "tensor",
        "size_average": "boolean",  # can be None (deprecated)
        "ignore_index": "integer",
        "reduce": "boolean",  # can be None (deprecated)
        "reduction": "string",
        "label_smoothing": "float"
    },
    "inner": {},
}
signatures["torch.nn.functional.dropout"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "p": "float",
        "training": "boolean",
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.dropout2d"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "p": "float",
        "training": "boolean",
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.dropout3d"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "p": "float",
        "training": "boolean",
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.elu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "alpha": "float",
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.embedding"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.feature_alpha_dropout"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "p": "float",
        "training": "boolean",
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.fold_1"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "dilation": "integer",
        "padding": "integer",
        "stride": "integer"
    },
    "inner": {}
}
signatures["torch.nn.functional.fold_2"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple",
        "kernel_size": "integer"
    },
    "kwargs": {
        "dilation": "integer",
        "padding": "integer",
        "stride": "integer"
    },
    "inner": {}
}
signatures["torch.nn.functional.fold_3"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "dilation": "tuple",
        "padding": "tuple",
        "stride": "tuple"
    },
    "inner": {}
}
signatures["torch.nn.functional.fold_4"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple",
        "kernel_size": "integer"
    },
    "kwargs": {
        "dilation": "tuple",
        "padding": "tuple",
        "stride": "tuple"
    },
    "inner": {}
}
signatures["torch.nn.functional.fold_5"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer",
        "kernel_size": "integer"
    },
    "kwargs": {
        "dilation": "integer",
        "padding": "integer",
        "stride": "integer"
    },
    "inner": {}
}
signatures["torch.nn.functional.fold_6"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "dilation": "integer",
        "padding": "integer",
        "stride": "integer"
    },
    "inner": {}
}
signatures["torch.nn.functional.fold_7"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer",
        "kernel_size": "integer"
    },
    "kwargs": {
        "dilation": "tuple",
        "padding": "tuple",
        "stride": "tuple"
    },
    "inner": {}
}
signatures["torch.nn.functional.fold_8"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "dilation": "tuple",
        "padding": "tuple",
        "stride": "tuple"
    },
    "inner": {}
}
signatures["torch.nn.functional.gelu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "approximate": "string"
    },
    "inner": {},
}
signatures["torch.nn.functional.glu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.group_norm"] = {
    "args": {
        "input": "tensor",
        "num_groups": "integer"
    },
    "kwargs": {
        "weight": "tensor",  # can be None
        "bias": "tensor",    # can be None
        "eps": "float"
    },
    "inner": {},
}
signatures["torch.nn.functional.gumbel_softmax"] = {
    "args": {
        "logits": "tensor"
    },
    "kwargs": {
        "tau": "float",
        "hard": "boolean",
        "eps": "float",
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.hardshrink"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "lambd": "float"  # could accept any number; using float
    },
    "inner": {},
}
signatures["torch.nn.functional.hardtanh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "min_val": "float",
        "max_val": "float",
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.hinge_embedding_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "margin": "float",
        "size_average": "boolean",  # can be None (deprecated)
        "reduce": "boolean",        # can be None (deprecated)
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.nn.functional.huber_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "reduction": "string",
        "delta": "float"
    },
    "inner": {},
}
signatures["torch.nn.functional.kl_div"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "size_average": "boolean",  # can be None (deprecated)
        "reduce": "boolean",        # can be None (deprecated)
        "reduction": "string",
        "log_target": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.leaky_relu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "negative_slope": "float",  # Scalar number; int also allowed but using float
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.linear"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor"  # can be None
    },
    "inner": {},
}
signatures["torch.nn.functional.local_response_norm"] = {
    "args": {
        "input": "tensor",
        "size": "integer"
    },
    "kwargs": {
        "alpha": "float",
        "beta": "float",
        "k": "float"
    },
    "inner": {},
}
signatures["torch.nn.functional.lp_pool1d_1"] = {
    "args": {
        "input": "tensor",
        "norm_type": "float",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "ceil_mode": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.lp_pool1d_2"] = {
    "args": {
        "input": "tensor",
        "norm_type": "float",
        "kernel_size": "tuple"  # tuple of one int
    },
    "kwargs": {
        "stride": "tuple",  # tuple of one int
        "ceil_mode": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.margin_ranking_loss"] = {
    "args": {
        "input1": "tensor",
        "input2": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "margin": "float",
        "size_average": "boolean",  # could also be None in practice
        "reduce": "boolean",        # could also be None in practice
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool1d_1"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",  # can be None or tuple
        "padding": "integer",  # can be tuple of length 1
        "dilation": "integer",  # can be tuple of length 1
        "ceil_mode": "boolean",
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool1d_2"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",  # can be None or integer
        "padding": "tuple",  # can be integer
        "dilation": "tuple",  # can be integer
        "ceil_mode": "boolean",
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_1"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "ceil_mode": "boolean",
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_2"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "tuple",
        "ceil_mode": "boolean",
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_3"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "tuple",
        "dilation": "integer",
        "ceil_mode": "boolean",
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_4"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "tuple",
        "dilation": "tuple",
        "ceil_mode": "boolean",
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_5"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "integer",
        "dilation": "integer",
        "ceil_mode": "boolean",
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_6"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "integer",
        "dilation": "tuple",
        "ceil_mode": "boolean",
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_7"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "integer",
        "ceil_mode": "boolean",
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_8"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "tuple",
        "ceil_mode": "boolean",
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_9"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "ceil_mode": "boolean",
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_10"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "tuple",
        "ceil_mode": "boolean",
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_11"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "tuple",
        "dilation": "integer",
        "ceil_mode": "boolean",
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_12"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "tuple",
        "dilation": "tuple",
        "ceil_mode": "boolean",
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_13"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "integer",
        "dilation": "integer",
        "ceil_mode": "boolean",
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_14"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "integer",
        "dilation": "tuple",
        "ceil_mode": "boolean",
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_15"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "integer",
        "ceil_mode": "boolean",
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_16"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "tuple",
        "ceil_mode": "boolean",
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool3d_1"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",  # can be None; treated as integer here
        "padding": "integer",
        "dilation": "integer",
        "ceil_mode": "boolean",
        "return_indices": "boolean"
    },
    "inner": {},
}

signatures["torch.nn.functional.max_pool3d_2"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",  # can be None; treated as tuple here
        "padding": "tuple",
        "dilation": "tuple",
        "ceil_mode": "boolean",
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.mish"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.mse_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "size_average": "boolean",  # can be None; using boolean per allowed types
        "reduce": "boolean",        # can be None; using boolean per allowed types
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.nn.functional.multi_margin_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "p": "integer",
        "margin": "float",
        "weight": "tensor",
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.nn.functional.multilabel_margin_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.nn.functional.multilabel_soft_margin_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "weight": "tensor",  # optional or None
        "size_average": "boolean",  # deprecated
        "reduce": "boolean",  # deprecated
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.nn.functional.normalize_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "p": "float",
        "dim": "integer",
        "eps": "float",
        "out": "tensor"
    },
    "inner": {},
}

signatures["torch.nn.functional.normalize_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "p": "float",
        "dim": "tuple",  # tuple of ints
        "eps": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.functional.one_hot"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "num_classes": "integer"  # can be None
    },
    "inner": {},
}
signatures["torch.nn.functional.pad_1"] = {
    "args": {
        "input": "tensor",
        "pad": "tuple"
    },
    "kwargs": {
        "mode": "string",
        "value": "float"  # could also accept int
    },
    "inner": {},
}
signatures["torch.nn.functional.pad_2"] = {
    "args": {
        "input": "tensor",
        "pad": "tuple"
    },
    "kwargs": {
        "mode": "string",
        "value": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.pairwise_distance"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {
        "p": "float",  # could also accept integer norm orders in practice
        "eps": "float",
        "keepdim": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.pdist"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "p": "float"  # numeric p-norm value; accepts floats including inf
    },
    "inner": {},
}
signatures["torch.nn.functional.pixel_shuffle"] = {
    "args": {
        "input": "tensor",
        "upscale_factor": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.pixel_unshuffle"] = {
    "args": {
        "input": "tensor",
        "downscale_factor": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.poisson_nll_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "log_input": "boolean",
        "full": "boolean",
        "eps": "float",
        "reduction": "string"
    },
    "inner": {}
}
signatures["torch.nn.functional.prelu"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"  # docs mention scalar or 1-D per-channel, but API expects a tensor
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.relu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.relu6"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.rrelu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "lower": "float",
        "upper": "float",
        "training": "boolean",
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.selu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.silu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.soft_margin_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "size_average": "boolean",  # can be None in older versions
        "reduce": "boolean",        # can be None in older versions
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.nn.functional.softmax"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"  # docs list default None, but practical usage expects an int
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.nn.functional.softmin"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.nn.functional.softplus"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "beta": "float",
        "threshold": "float"
    },
    "inner": {},
}
signatures["torch.nn.functional.softshrink"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "lambd": "float"
    },
    "inner": {},
}
signatures["torch.nn.functional.softsign"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.tanhshrink"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.threshold"] = {
    "args": {
        "input": "tensor",
        "threshold": "float",
        "value": "float"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.triplet_margin_loss"] = {
    "args": {
        "anchor": "tensor",
        "positive": "tensor",
        "negative": "tensor"
    },
    "kwargs": {
        "margin": "float",
        "p": "float",  # could also accept integer values in practice
        "eps": "float",
        "swap": "boolean",
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {}
}
signatures["torch.nn.init.constant__1"] = {
    "args": {
        "tensor": "tensor",
        "val": "float"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.init.constant__2"] = {
    "args": {
        "tensor": "tensor",
        "val": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.init.dirac_"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "groups": "integer"
    },
    "inner": {},
}
signatures["torch.nn.init.eye_"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.init.kaiming_normal_"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "a": "float",
        "mode": "string",
        "nonlinearity": "string"
    },
    "inner": {},
}
signatures["torch.nn.init.kaiming_uniform_"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "a": "float",
        "mode": "string",
        "nonlinearity": "string"
    },
    "inner": {},
}
signatures["torch.nn.init.normal_"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "mean": "float",
        "std": "float"
    },
    "inner": {},
}
signatures["torch.nn.init.ones_"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.init.orthogonal_"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "gain": "float"
    },
    "inner": {},
}
signatures["torch.nn.init.uniform_"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "a": "float",
        "b": "float"
    },
    "inner": {}
}
signatures["torch.nn.init.xavier_normal_"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "gain": "float"  # could be int-compatible, but treated as float
    },
    "inner": {},
}
signatures["torch.nn.init.xavier_uniform_"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "gain": "float"
    },
    "inner": {},
}
signatures["torch.nn.init.zeros_"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.utils.clip_grad_norm__1"] = {
    "args": {
        "parameters": "tensor",
        "max_norm": "float"
    },
    "kwargs": {
        "norm_type": "float",
        "error_if_nonfinite": "boolean",
        "foreach": "boolean"  # can also be None
    },
    "inner": {},
}
signatures["torch.nn.utils.clip_grad_norm__2"] = {
    "args": {
        "parameters": "tensor",
        "max_norm": "float"
    },
    "kwargs": {
        "norm_type": "integer",
        "error_if_nonfinite": "boolean",
        "foreach": "boolean"  # can also be None
    },
    "inner": {},
}
signatures["torch.nn.utils.clip_grad_norm__3"] = {
    "args": {
        "parameters": "tensor",
        "max_norm": "float"
    },
    "kwargs": {
        "norm_type": "string",  # typically "inf"
        "error_if_nonfinite": "boolean",
        "foreach": "boolean"  # can also be None
    },
    "inner": {},
}
signatures["torch.nn.utils.clip_grad_norm__4"] = {
    "args": {
        "parameters": "tensor",
        "max_norm": "integer"
    },
    "kwargs": {
        "norm_type": "float",
        "error_if_nonfinite": "boolean",
        "foreach": "boolean"  # can also be None
    },
    "inner": {},
}
signatures["torch.nn.utils.clip_grad_norm__5"] = {
    "args": {
        "parameters": "tensor",
        "max_norm": "integer"
    },
    "kwargs": {
        "norm_type": "integer",
        "error_if_nonfinite": "boolean",
        "foreach": "boolean"  # can also be None
    },
    "inner": {},
}
signatures["torch.nn.utils.clip_grad_norm__6"] = {
    "args": {
        "parameters": "tensor",
        "max_norm": "integer"
    },
    "kwargs": {
        "norm_type": "string",  # typically "inf"
        "error_if_nonfinite": "boolean",
        "foreach": "boolean"  # can also be None
    },
    "inner": {},
}
signatures["torch.nn.utils.clip_grad_norm__7"] = {
    "args": {
        "parameters": "tensor_list",
        "max_norm": "float"
    },
    "kwargs": {
        "norm_type": "float",
        "error_if_nonfinite": "boolean",
        "foreach": "boolean"  # can also be None
    },
    "inner": {},
}
signatures["torch.nn.utils.clip_grad_norm__8"] = {
    "args": {
        "parameters": "tensor_list",
        "max_norm": "float"
    },
    "kwargs": {
        "norm_type": "integer",
        "error_if_nonfinite": "boolean",
        "foreach": "boolean"  # can also be None
    },
    "inner": {},
}
signatures["torch.nn.utils.clip_grad_norm__9"] = {
    "args": {
        "parameters": "tensor_list",
        "max_norm": "float"
    },
    "kwargs": {
        "norm_type": "string",  # typically "inf"
        "error_if_nonfinite": "boolean",
        "foreach": "boolean"  # can also be None
    },
    "inner": {},
}
signatures["torch.nn.utils.clip_grad_norm__10"] = {
    "args": {
        "parameters": "tensor_list",
        "max_norm": "integer"
    },
    "kwargs": {
        "norm_type": "float",
        "error_if_nonfinite": "boolean",
        "foreach": "boolean"  # can also be None
    },
    "inner": {},
}
signatures["torch.nn.utils.clip_grad_norm__11"] = {
    "args": {
        "parameters": "tensor_list",
        "max_norm": "integer"
    },
    "kwargs": {
        "norm_type": "integer",
        "error_if_nonfinite": "boolean",
        "foreach": "boolean"  # can also be None
    },
    "inner": {},
}
signatures["torch.nn.utils.clip_grad_norm__12"] = {
    "args": {
        "parameters": "tensor_list",
        "max_norm": "integer"
    },
    "kwargs": {
        "norm_type": "string",  # typically "inf"
        "error_if_nonfinite": "boolean",
        "foreach": "boolean"  # can also be None
    },
    "inner": {},
}
signatures["torch.nn.utils.parameters_to_vector"] = {
    "args": {
        "parameters": "tensor_list"  # Iterable of tensors
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.utils.rnn.pack_padded_sequence_1"] = {
    "args": {
        "input": "tensor",
        "lengths": "tensor"  # 1D int tensor
    },
    "kwargs": {
        "batch_first": "boolean",
        "enforce_sorted": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.utils.rnn.pack_padded_sequence_2"] = {
    "args": {
        "input": "tensor",
        "lengths": "list"  # list of integers
    },
    "kwargs": {
        "batch_first": "boolean",
        "enforce_sorted": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.utils.rnn.pack_sequence"] = {
    "args": {
        "sequences": "tensor_list"
    },
    "kwargs": {
        "enforce_sorted": "boolean"
    },
    "inner": {},
}
signatures["torch.special.digamma"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.entr"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.erf"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.erfc"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.erfcx"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.erfinv"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.exp2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.expit"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.expm1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.gammainc"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.gammaincc"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.gammaln"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.i0"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.i0e"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.i1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.i1e"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.log1p"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.log_softmax"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "dtype": "dtype"  # optional cast dtype before computation
    },
    "inner": {},
}
signatures["torch.special.logit"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "eps": "float",  # can be None as well
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.logsumexp_1"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "dtype": "dtype",
        "out": "tensor"  # unsure if supported in special.*, but many ops accept it
    },
    "inner": {},
}
signatures["torch.special.logsumexp_2"] = {
    "args": {
        "input": "tensor",
        "dim": "tuple"
    },
    "kwargs": {
        "keepdim": "boolean",
        "dtype": "dtype",
        "out": "tensor"  # unsure if supported in special.*, but many ops accept it
    },
    "inner": {},
}
signatures["torch.special.ndtr"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.polygamma"] = {
    "args": {
        "n": "integer",
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.psi"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.round"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "decimals": "integer",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.sinc"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.xlogy_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.xlogy_2"] = {
    "args": {
        "input": "tensor",
        "other": "float"  # accepts Python number; using float as closest match
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.xlogy_3"] = {
    "args": {
        "input": "tensor",
        "other": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.xlogy_4"] = {
    "args": {
        "input": "float",  # accepts Python number; using float as closest match
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.xlogy_5"] = {
    "args": {
        "input": "integer",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.zeta_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.zeta_2"] = {
    "args": {
        "input": "tensor",
        "other": "float"  # could also accept integer scalars, but using float
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.zeta_3"] = {
    "args": {
        "input": "float",  # could also accept integer scalars, but using float
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.zeta_4"] = {
    "args": {
        "input": "float",  # could also accept integer scalars, but using float
        "other": "float"   # could also accept integer scalars, but using float
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.DoubleStorage_1"] = {
    "args": {
        "size": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.DoubleStorage_2"] = {
    "args": {
        "data": "list"  # could also accept other buffer-like sequences; using list as closest match
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.abs_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.acos"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # optional
    },
    "inner": {},
}
signatures["torch.acosh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.addcdiv_1"] = {
    "args": {
        "input": "tensor",
        "tensor1": "tensor",
        "tensor2": "tensor"
    },
    "kwargs": {
        "value": "integer",  # Number: integer allowed
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.addcdiv_2"] = {
    "args": {
        "input": "tensor",
        "tensor1": "tensor",
        "tensor2": "tensor"
    },
    "kwargs": {
        "value": "float",  # Number: float allowed
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.addcmul_1"] = {
    "args": {
        "input": "tensor",
        "tensor1": "tensor",
        "tensor2": "tensor"
    },
    "kwargs": {
        "value": "float",  # could also be integer depending on dtypes
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.addcmul_2"] = {
    "args": {
        "input": "tensor",
        "tensor1": "tensor",
        "tensor2": "tensor"
    },
    "kwargs": {
        "value": "integer",  # could also be float depending on dtypes
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.addmm_1"] = {
    "args": {
        "input": "tensor",
        "mat1": "tensor",
        "mat2": "tensor",
        "out_dtype": "dtype"  # optional in API, but kept as positional per docs
    },
    "kwargs": {
        "beta": "float",
        "alpha": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.addmm_2"] = {
    "args": {
        "input": "tensor",
        "mat1": "tensor",
        "mat2": "tensor",
        "out_dtype": "dtype"  # optional in API, but kept as positional per docs
    },
    "kwargs": {
        "beta": "integer",  # could also be float depending on input dtype
        "alpha": "integer",  # could also be float depending on input dtype
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.addmv"] = {
    "args": {
        "input": "tensor",
        "mat": "tensor",
        "vec": "tensor"
    },
    "kwargs": {
        "beta": "float",  # number; integer allowed for integer dtypes
        "alpha": "float",  # number; integer allowed for integer dtypes
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.amax_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer",
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.amax_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "tuple",
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.amax_3"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        # dim can be None (reduce over all dims); representing by omitting "dim"
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.amin_1"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.amin_2"] = {
    "args": {
        "input": "tensor",
        "dim": "tuple"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.amin_3"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.are_deterministic_algorithms_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.as_strided_1"] = {
    "args": {
        "input": "tensor",
        "size": "tuple",
        "stride": "tuple"
    },
    "kwargs": {
        "storage_offset": "integer"
    },
    "inner": {},
}
signatures["torch.as_strided_2"] = {
    "args": {
        "input": "tensor",
        "size": "tuple",
        "stride": "list"
    },
    "kwargs": {
        "storage_offset": "integer"
    },
    "inner": {},
}
signatures["torch.as_strided_3"] = {
    "args": {
        "input": "tensor",
        "size": "list",
        "stride": "tuple"
    },
    "kwargs": {
        "storage_offset": "integer"
    },
    "inner": {},
}
signatures["torch.as_strided_4"] = {
    "args": {
        "input": "tensor",
        "size": "list",
        "stride": "list"
    },
    "kwargs": {
        "storage_offset": "integer"
    },
    "inner": {},
}
signatures["torch.as_tensor_1"] = {
    "args": {
        "data": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.as_tensor_2"] = {
    "args": {
        "data": "list"  # numpy arrays also accepted; mapped to list
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.as_tensor_3"] = {
    "args": {
        "data": "tuple"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.as_tensor_4"] = {
    "args": {
        "data": "integer"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.as_tensor_5"] = {
    "args": {
        "data": "float"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.as_tensor_6"] = {
    "args": {
        "data": "boolean"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.asarray_1"] = {
    "args": {
        "obj": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "copy": "boolean",
        "requires_grad": "boolean"
    },
    "inner": {},
}

signatures["torch.asarray_2"] = {
    "args": {
        "obj": "list"  # also used to approximate NumPy arrays/buffer-like inputs
    },
    "kwargs": {
        "dtype": "dtype",
        "copy": "boolean",
        "requires_grad": "boolean"
    },
    "inner": {},
}

signatures["torch.asarray_3"] = {
    "args": {
        "obj": "tuple"  # sequence of scalars
    },
    "kwargs": {
        "dtype": "dtype",
        "copy": "boolean",
        "requires_grad": "boolean"
    },
    "inner": {},
}

signatures["torch.asarray_4"] = {
    "args": {
        "obj": "integer"  # scalar
    },
    "kwargs": {
        "dtype": "dtype",
        "copy": "boolean",
        "requires_grad": "boolean"
    },
    "inner": {},
}

signatures["torch.asarray_5"] = {
    "args": {
        "obj": "float"  # scalar
    },
    "kwargs": {
        "dtype": "dtype",
        "copy": "boolean",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.asin"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.asinh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.atan"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.atan2"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.atanh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.atleast_1d"] = {
    "args": {
        "tensors": "tensor_list"  # varargs: one or more tensors
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.atleast_2d"] = {
    "args": {
        "tensors": "tensor_list"  # varargs: one or more tensors
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.atleast_3d_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.atleast_3d_2"] = {
    "args": {
        "tensors": "tensor_list"  # varargs of tensors
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.baddbmm_1"] = {
    "args": {
        "input": "tensor",
        "batch1": "tensor",
        "batch2": "tensor",
        "out_dtype": "dtype"  # optional; only supported in specific cases per docs
    },
    "kwargs": {
        "beta": "float",   # number; using float variant
        "alpha": "float",  # number; using float variant
        "out": "tensor"
    },
    "inner": {},
}

signatures["torch.baddbmm_2"] = {
    "args": {
        "input": "tensor",
        "batch1": "tensor",
        "batch2": "tensor",
        "out_dtype": "dtype"  # optional; only supported in specific cases per docs
    },
    "kwargs": {
        "beta": "integer",  # number; integer variant for integer tensors
        "alpha": "integer", # number; integer variant for integer tensors
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.bincount"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "weights": "tensor",
        "minlength": "integer"
    },
    "inner": {},
}
signatures["torch.bitwise_not"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.bmm"] = {
    "args": {
        "input": "tensor",
        "mat2": "tensor"
    },
    "kwargs": {
        "out_dtype": "dtype",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.broadcast_shapes_1"] = {
    "args": {
        "shapes": "tuple"  # varargs: multiple tuples of ints (sizes)
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.broadcast_shapes_2"] = {
    "args": {
        "shapes": "list"  # varargs: multiple lists of ints (sizes)
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.bucketize_1"] = {
    "args": {
        "input": "tensor",
        "boundaries": "tensor"
    },
    "kwargs": {
        "out_int32": "boolean",
        "right": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.bucketize_2"] = {
    "args": {
        "input": "integer",  # scalar integer
        "boundaries": "tensor"
    },
    "kwargs": {
        "out_int32": "boolean",
        "right": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.bucketize_3"] = {
    "args": {
        "input": "float",  # scalar float
        "boundaries": "tensor"
    },
    "kwargs": {
        "out_int32": "boolean",
        "right": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.ceil"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.clamp_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "min": "float",  # Number -> chose float; ints also valid in practice
        "max": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.clamp_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "min": "tensor",
        "max": "tensor",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.clamp_3"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "min": "float",  # Number -> chose float; ints also valid in practice
        "max": "tensor",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.clamp_4"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "min": "tensor",
        "max": "float",  # Number -> chose float; ints also valid in practice
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.clamp_max_1"] = {
    "args": {
        "input": "tensor",
        "max": "float"  # could also be integer; provided as separate signature
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.clamp_max_2"] = {
    "args": {
        "input": "tensor",
        "max": "integer"  # could also be float; provided as separate signature
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.clamp_max_3"] = {
    "args": {
        "input": "tensor",
        "max": "tensor"  # tensor max is elementwise
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.conj"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.conj_physical"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.cos"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.cosh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # optional
    },
    "inner": {},
}
signatures["torch.dequantize_1"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.dequantize_2"] = {
    "args": {
        "tensors": "tensor_list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.digamma"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.eq_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.eq_2"] = {
    "args": {
        "input": "tensor",
        "other": "float"  # Could also accept integers, but docs specify "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.equal"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.erf"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.erfc"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.exp"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.exp2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.expm1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.fake_quantize_per_tensor_affine"] = {
    "args": {
        "input": "tensor",
        "scale": "float",
        "zero_point": "integer",
        "quant_min": "integer",
        "quant_max": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.floor"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.fmax"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.fmin"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.ge_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.ge_2"] = {
    "args": {
        "input": "tensor",
        "other": "float"  # 'other' can be any number; using float to represent numeric
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.get_default_dtype"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.get_num_threads"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.gt_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}

signatures["torch.gt_2"] = {
    "args": {
        "input": "tensor",
        "other": "float"  # could also accept integer in practice, but docs specify float
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.heaviside"] = {
    "args": {
        "input": "tensor",
        "values": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.histc_1"] = {
    "args": {
        "input": "tensor",
        "bins": "integer",
        "min": "float",  # Scalar; could also be integer
        "max": "float"   # Scalar; could also be integer
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.histc_2"] = {
    "args": {
        "input": "tensor",
        "bins": "integer",
        "min": "integer",  # Scalar; could also be float
        "max": "integer"   # Scalar; could also be float
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.hypot"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.i0"] = {
    "args": {
        "input": "tensor"  # may accept number-like inputs but documented as Tensor
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.igammac"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.is_floating_point_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_floating_point_2"] = {
    "args": {
        "input": "dtype"  # Some versions may not support dtype directly; tensor is the common case.
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_grad_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_inference_mode_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_storage"] = {
    "args": {
        "obj": "tensor"  # any Python object; checks if it's a Storage
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_tensor"] = {
    "args": {
        "obj": "tensor"  # Accepts any object; using "tensor" as closest match
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_warn_always_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.isneginf"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # optional output tensor
    },
    "inner": {},
}
signatures["torch.lcm"] = {
    "args": {
        "input": "tensor",  # expects integer tensor
        "other": "tensor"   # expects integer tensor
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.le_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.le_2"] = {
    "args": {
        "input": "tensor",
        "other": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.le_3"] = {
    "args": {
        "input": "tensor",
        "other": "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.le_4"] = {
    "args": {
        "input": "tensor",
        "other": "boolean"  # Scalar likely includes bool; included for completeness
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.lerp_1"] = {
    "args": {
        "input": "tensor",
        "end": "tensor",
        "weight": "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.lerp_2"] = {
    "args": {
        "input": "tensor",
        "end": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.lgamma"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.log10"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.log1p"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.log2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.logaddexp"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.logaddexp2"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.lt_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}

signatures["torch.lt_2"] = {
    "args": {
        "input": "tensor",
        "other": "float"  # In practice, PyTorch accepts any number (int/float); docs mention float.
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.lu_unpack"] = {
    "args": {
        "LU_data": "tensor",
        "LU_pivots": "tensor"
    },
    "kwargs": {
        "unpack_data": "boolean",
        "unpack_pivots": "boolean"
    },
    "inner": {},
}
signatures["torch.masked_select"] = {
    "args": {
        "input": "tensor",
        "mask": "tensor"  # bool or byte tensor; using "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.maximum"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.minimum"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.mm"] = {
    "args": {
        "input": "tensor",
        "mat2": "tensor",
        "out_dtype": "dtype"  # optional; dtype of output (CUDA-specific behavior)
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.ne_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}

signatures["torch.ne_2"] = {
    "args": {
        "input": "tensor",
        "other": "float"  # doc says "number"; using float here
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}

signatures["torch.ne_3"] = {
    "args": {
        "input": "tensor",
        "other": "integer"  # doc says "number"; using integer here
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.neg"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # optional
    },
    "inner": {},
}
signatures["torch.nextafter"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.Hardsigmoid"] = {
    "args": {},
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.Hardswish"] = {
    "args": {},
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.LazyBatchNorm1d"] = {
    "args": {},
    "kwargs": {
        "eps": "float",
        "momentum": "float",  # Optional[float]; None is also allowed in PyTorch
        "affine": "boolean",
        "track_running_stats": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.LazyInstanceNorm1d"] = {
    "args": {},
    "kwargs": {
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.LazyInstanceNorm2d"] = {
    "args": {},
    "kwargs": {
        "eps": "float",
        "momentum": "float",  # could be None in some versions
        "affine": "boolean",
        "track_running_stats": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Parameter_1"] = {
    "args": {
        "data": "tensor"  # can be None
    },
    "kwargs": {
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.Parameter_2"] = {
    "args": {
    },
    "kwargs": {
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.hardsigmoid"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.hardswish"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nonzero"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor",
        "as_tuple": "boolean"
    },
    "inner": {},
}
signatures["torch.numel"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.polygamma_1"] = {
    "args": {
        "n": "integer",  # likely restricted to non-negative integers
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.polygamma_2"] = {
    "args": {
        "n": "tensor",  # some versions may allow tensor n (broadcastable); if not, use integer
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.positive"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.pow_1"] = {
    "args": {
        "input": "tensor",
        "exponent": "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.pow_2"] = {
    "args": {
        "input": "tensor",
        "exponent": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.pow_3"] = {
    "args": {
        "self": "float",  # PyTorch also accepts int, but docs specify float
        "exponent": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.promote_types"] = {
    "args": {
        "type1": "dtype",
        "type2": "dtype"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.ravel"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.real"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.reciprocal"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.reciprocal_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.relu_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_1"] = {
    "args": {
        "tensor1": "tensor",
        "tensor2": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_2"] = {
    "args": {
        "tensor1": "tensor",
        "tensor2": "dtype"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_3"] = {
    "args": {
        "tensor1": "tensor",
        "tensor2": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_4"] = {
    "args": {
        "tensor1": "tensor",
        "tensor2": "float"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_5"] = {
    "args": {
        "tensor1": "tensor",
        "tensor2": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_6"] = {
    "args": {
        "tensor1": "dtype",
        "tensor2": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_7"] = {
    "args": {
        "tensor1": "dtype",
        "tensor2": "dtype"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_8"] = {
    "args": {
        "tensor1": "dtype",
        "tensor2": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_9"] = {
    "args": {
        "tensor1": "dtype",
        "tensor2": "float"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_10"] = {
    "args": {
        "tensor1": "dtype",
        "tensor2": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_11"] = {
    "args": {
        "tensor1": "integer",
        "tensor2": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_12"] = {
    "args": {
        "tensor1": "integer",
        "tensor2": "dtype"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_13"] = {
    "args": {
        "tensor1": "integer",
        "tensor2": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_14"] = {
    "args": {
        "tensor1": "integer",
        "tensor2": "float"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_15"] = {
    "args": {
        "tensor1": "integer",
        "tensor2": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_16"] = {
    "args": {
        "tensor1": "float",
        "tensor2": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_17"] = {
    "args": {
        "tensor1": "float",
        "tensor2": "dtype"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_18"] = {
    "args": {
        "tensor1": "float",
        "tensor2": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_19"] = {
    "args": {
        "tensor1": "float",
        "tensor2": "float"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_20"] = {
    "args": {
        "tensor1": "float",
        "tensor2": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_21"] = {
    "args": {
        "tensor1": "boolean",
        "tensor2": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_22"] = {
    "args": {
        "tensor1": "boolean",
        "tensor2": "dtype"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_23"] = {
    "args": {
        "tensor1": "boolean",
        "tensor2": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_24"] = {
    "args": {
        "tensor1": "boolean",
        "tensor2": "float"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_25"] = {
    "args": {
        "tensor1": "boolean",
        "tensor2": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.roll_1"] = {
    "args": {
        "input": "tensor",
        "shifts": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.roll_2"] = {
    "args": {
        "input": "tensor",
        "shifts": "integer"
    },
    "kwargs": {
        "dims": "integer"
    },
    "inner": {},
}
signatures["torch.roll_3"] = {
    "args": {
        "input": "tensor",
        "shifts": "integer"
    },
    "kwargs": {
        "dims": "tuple"
    },
    "inner": {},
}
signatures["torch.roll_4"] = {
    "args": {
        "input": "tensor",
        "shifts": "tuple"
    },
    "kwargs": {
        "dims": "tuple"
    },
    "inner": {},
}
signatures["torch.round"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "decimals": "integer",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.rsqrt"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.set_flush_denormal"] = {
    "args": {
        "mode": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_num_interop_threads"] = {
    "args": {
        "num_threads": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_warn_always"] = {
    "args": {
        "enabled": "boolean"  # could be named 'mode' or 'b' in some versions
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.sgn"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.sigmoid"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.sigmoid_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.sign"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.signbit"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.sin"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.sinh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.split_1"] = {
    "args": {
        "tensor": "tensor",
        "split_size_or_sections": "integer"
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.split_2"] = {
    "args": {
        "tensor": "tensor",
        "split_size_or_sections": "list"  # could also accept tuple in practice, but docs specify list
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.sqrt"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.squeeze_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.squeeze_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "tuple"  # tuple of ints
    },
    "inner": {},
}
signatures["torch.squeeze_3"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "list"  # list of ints (also accepted)
    },
    "inner": {},
}
signatures["torch.squeeze_4"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.tan"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.tanh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.tril"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "diagonal": "integer",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.triu"] = {
    "args": {
        "input": "tensor",
        "diagonal": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.trunc"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.trunc_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.typename"] = {
    "args": {
        "obj": "tensor"  # could be any object; using 'tensor' as closest allowed type
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.unique"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "sorted": "boolean",
        "return_inverse": "boolean",
        "return_counts": "boolean",
        "dim": "integer"  # optional, can be None
    },
    "inner": {},
}
signatures["torch.unique_consecutive"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "return_inverse": "boolean",
        "return_counts": "boolean",
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.view_as_complex"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
