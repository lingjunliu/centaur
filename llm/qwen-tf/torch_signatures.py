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
