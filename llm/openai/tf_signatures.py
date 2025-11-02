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
signatures["tf.raw_ops.SparseSegmentSum"] = {
    "args": {
        "data": "tensor",
        "indices": "tensor",
        "segment_ids": "tensor"
    },
    "kwargs": {
        "sparse_gradient": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.zeros_1"] = {
    "args": {
        "shape": "list"
    },
    "kwargs": {
        "dtype": "dtype",
        "name": "string",
        "layout": "tensor"  # dtensor.Layout object
    },
    "inner": {}
}
signatures["tf.zeros_2"] = {
    "args": {
        "shape": "tuple"
    },
    "kwargs": {
        "dtype": "dtype",
        "name": "string",
        "layout": "tensor"  # dtensor.Layout object
    },
    "inner": {}
}
signatures["tf.zeros_3"] = {
    "args": {
        "shape": "tensor"  # 1-D Tensor of int32
    },
    "kwargs": {
        "dtype": "dtype",
        "name": "string",
        "layout": "tensor"  # dtensor.Layout object
    },
    "inner": {}
}
signatures["tf.experimental.numpy.compress"] = {
    "args": {
        "condition": "tensor",  # boolean tensor
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer"  # can be None
    },
    "inner": {}
}
signatures["tf.image.adjust_hue"] = {
    "args": {
        "image": "tensor",
        "delta": "float"  # Could also be a scalar tensor; using float per docs
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.selu"] = {
    "args": {
        "features": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel"] = {
    "args": {
        "inputs": "tensor",
        "min": "tensor",
        "max": "tensor"
    },
    "kwargs": {
        "num_bits": "integer",
        "narrow_range": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.stateless_random_crop"] = {
    "args": {
        "value": "tensor",
        "size": "tensor",  # Also accepts list/tuple of ints; TF converts to Tensor
        "seed": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.signal.fft2d"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.Graph"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["tf.ones_1"] = {
    "args": {
        "shape": "list"
    },
    "kwargs": {
        "dtype": "dtype",
        "name": "string",
        "layout": "tensor"  # DTensor Layout object
    },
    "inner": {}
}
signatures["tf.ones_2"] = {
    "args": {
        "shape": "tuple"
    },
    "kwargs": {
        "dtype": "dtype",
        "name": "string",
        "layout": "tensor"  # DTensor Layout object
    },
    "inner": {}
}
signatures["tf.ones_3"] = {
    "args": {
        "shape": "tensor"  # 1-D int32 tensor
    },
    "kwargs": {
        "dtype": "dtype",
        "name": "string",
        "layout": "tensor"  # DTensor Layout object
    },
    "inner": {}
}
signatures["tf.experimental.numpy.rot90"] = {
    "args": {
        "m": "tensor"
    },
    "kwargs": {
        "k": "integer",
        "axes": "tuple"  # tuple of two integers
    },
    "inner": {}
}
signatures["tf.image.flip_left_right"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {
        # "name": "string"  # Usually exists in TF APIs, but not described in the provided doc.
    },
    "inner": {}
}
signatures["tf.signal.rfft2d"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "fft_length": "list",  # Tensor of type int32 and shape [2] is a list of 2 integers
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.l2_loss"] = {
    "args": {
        "t": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.bessel_i0e"] = {
    "args": {
        "x": "tensor"  # Tensor or SparseTensor
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.DrawBoundingBoxes"] = {
    "args": {
        "images": "tensor",
        "boxes": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
