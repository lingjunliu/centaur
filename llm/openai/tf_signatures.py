signatures = {}
signatures["tf.raw_ops.GreaterEqual"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.adjust_saturation_1"] = {
    "args": {
        "image": "tensor",
        "saturation_factor": "float"  # could also be a scalar tensor
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}

signatures["tf.image.adjust_saturation_2"] = {
    "args": {
        "image": "tensor",
        "saturation_factor": "tensor"  # scalar tensor
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.feature_column.categorical_column_with_identity"] = {
    "args": {
        "key": "string",
        "num_buckets": "integer"
    },
    "kwargs": {
        "default_value": "integer"  # Optional; None allowed, but when set must be an integer in [0, num_buckets)
    },
    "inner": {}
}
signatures["tf.io.serialize_tensor"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.atan2"] = {
    "args": {
        "y": "tensor",
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Empty"] = {
    "args": {
        "shape": "tensor",  # 1-D int32 Tensor representing shape
        "dtype": "dtype"
    },
    "kwargs": {
        "init": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.SparseReduceSumSparse"] = {
    "args": {
        "input_indices": "tensor",
        "input_values": "tensor",
        "input_shape": "tensor",
        "reduction_axes": "tensor"
    },
    "kwargs": {
        "keep_dims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.isfinite"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.raw_ops.Relu"] = {
    "args": {
        "features": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.sysconfig.get_include"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["tf.get_static_value"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "partial": "boolean"
    },
    "inner": {}
}
signatures["tf.compat.path_to_str"] = {
    "args": {
        "path": "string"  # Also accepts os.PathLike; mapped to string
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.feature_column.sequence_categorical_column_with_hash_bucket"] = {
    "args": {
        "key": "string",
        "hash_bucket_size": "integer"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.identity"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.ComputeAccidentalHits"] = {
    "args": {
        "true_classes": "tensor",
        "sampled_candidates": "tensor",
        "num_true": "integer"
    },
    "kwargs": {
        "seed": "integer",
        "seed2": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Real"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "Tout": "dtype",
        "name": "string"
    },
    "inner": {}
}
