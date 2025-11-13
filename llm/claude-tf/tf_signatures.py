signatures = {}
signatures["tf.broadcast_dynamic_shape"] = {
    "args": {
        "shape_x": "tensor",
        "shape_y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.compat.forward_compatible"] = {
    "args": {
        "year": "integer",
        "month": "integer",
        "day": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.convert_to_tensor"] = {
    "args": {
        "value": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "dtype_hint": "dtype",
        "name": "string"
    },
    "inner": {}
}

signatures["tf.convert_to_tensor_2"] = {
    "args": {
        "value": "list"
    },
    "kwargs": {
        "dtype": "dtype",
        "dtype_hint": "dtype",
        "name": "string"
    },
    "inner": {}
}

signatures["tf.convert_to_tensor_3"] = {
    "args": {
        "value": "integer"  # Python scalars
    },
    "kwargs": {
        "dtype": "dtype",
        "dtype_hint": "dtype",
        "name": "string"
    },
    "inner": {}
}

signatures["tf.convert_to_tensor_4"] = {
    "args": {
        "value": "float"  # Python scalars
    },
    "kwargs": {
        "dtype": "dtype",
        "dtype_hint": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.Counter"] = {
    "args": {},
    "kwargs": {
        "start": "integer",
        "step": "integer",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.data.experimental.from_variant"] = {
    "args": {
        "variant": "tensor",
        "structure": "tuple"  # (nested) structure of tf.TypeSpec objects
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.append"] = {
    "args": {
        "arr": "tensor",
        "values": "tensor"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.argmin"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.bitwise_not"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.conj"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.cumsum"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.diag"] = {
    "args": {
        "v": "tensor"
    },
    "kwargs": {
        "k": "integer"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.dot"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.fix"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.floor_divide"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.isfinite"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.isinf"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.isnan"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.isposinf"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.kron"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
# Based on NumPy's moveaxis documentation, source and destination can be integers or sequences of integers
# Creating multiple signatures for different type combinations

signatures["tf.experimental.numpy.moveaxis_1"] = {
    "args": {
        "a": "tensor",
        "source": "integer",
        "destination": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["tf.experimental.numpy.moveaxis_2"] = {
    "args": {
        "a": "tensor",
        "source": "list",
        "destination": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["tf.experimental.numpy.moveaxis_3"] = {
    "args": {
        "a": "tensor",
        "source": "tuple",
        "destination": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.nanprod"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "dtype": "dtype",
        "keepdims": "boolean"
    },
    "inner": {}
}

signatures["tf.experimental.numpy.nanprod_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "dtype": "dtype",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.promote_types"] = {
    "args": {
        "type1": "dtype",
        "type2": "dtype"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.ravel"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.signbit"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.tril"] = {
    "args": {
        "m": "tensor"
    },
    "kwargs": {
        "k": "integer"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.vdot"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.identity_n"] = {
    "args": {
        "input": "tensor_list"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.central_crop"] = {
    "args": {
        "image": "tensor",
        "central_fraction": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.crop_to_bounding_box"] = {
    "args": {
        "image": "tensor",
        "offset_height": "integer",
        "offset_width": "integer",
        "target_height": "integer",
        "target_width": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.grayscale_to_rgb"] = {
    "args": {
        "images": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.random_contrast"] = {
    "args": {
        "image": "tensor",
        "lower": "float",
        "upper": "float"
    },
    "kwargs": {
        "seed": "integer"
    },
    "inner": {}
}
signatures["tf.image.random_hue"] = {
    "args": {
        "image": "tensor",
        "max_delta": "float"
    },
    "kwargs": {
        "seed": "integer"
    },
    "inner": {}
}
signatures["tf.image.resize_with_crop_or_pad"] = {
    "args": {
        "image": "tensor",
        "target_height": "integer",
        "target_width": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.rgb_to_hsv"] = {
    "args": {
        "images": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.stateless_random_flip_left_right_1"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {
        "seed": "tuple"
    },
    "inner": {}
}

signatures["tf.image.stateless_random_flip_left_right_2"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {
        "seed": "tensor"
    },
    "inner": {}
}
signatures["tf.image.stateless_random_flip_up_down"] = {
    "args": {
        "image": "tensor",
        "seed": "tuple"  # seed can be tuple or tensor of shape [2]
    },
    "kwargs": {},
    "inner": {}
}

signatures["tf.image.stateless_random_flip_up_down_2"] = {
    "args": {
        "image": "tensor",
        "seed": "tensor"  # seed as Tensor of shape [2]
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.transpose"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.IndexedSlices"] = {
    "args": {
        "values": "tensor",
        "indices": "tensor"
    },
    "kwargs": {
        "dense_shape": "tensor"
    },
    "inner": {}
}
signatures["tf.linalg.eigh"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.LinearOperatorHouseholder"] = {
    "args": {
        "reflection_axis": "tensor"
    },
    "kwargs": {
        "is_non_singular": "boolean",
        "is_self_adjoint": "boolean",
        "is_positive_definite": "boolean",
        "is_square": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.LinearOperatorZeros"] = {
    "args": {
        "num_rows": "integer"
    },
    "kwargs": {
        "num_columns": "integer",
        "batch_shape": "list",
        "dtype": "dtype",
        "is_non_singular": "boolean",
        "is_self_adjoint": "boolean",
        "is_positive_definite": "boolean",
        "is_square": "boolean",
        "assert_proper_shapes": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.lstsq"] = {
    "args": {
        "matrix": "tensor",
        "rhs": "tensor"
    },
    "kwargs": {
        "l2_regularizer": "float",
        "fast": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.matrix_transpose"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "name": "string",
        "conjugate": "boolean"
    },
    "inner": {}
}
signatures["tf.linalg.solve"] = {
    "args": {
        "matrix": "tensor",
        "rhs": "tensor"
    },
    "kwargs": {
        "adjoint": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.trace"] = {
    "args": {
        "x": "tensor"
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
signatures["tf.math.bessel_i1e"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.cos"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.cumprod"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "exclusive": "boolean",
        "reverse": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.erf"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.floormod"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.invert_permutation"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.is_inf"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.lgamma"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.log1p"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.logical_not"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.maximum"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.polygamma"] = {
    "args": {
        "a": "tensor",
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.pow"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.real"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.special.bessel_j1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.squared_difference"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.zero_fraction"] = {
    "args": {
        "value": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nest.assert_same_structure_1"] = {
    "args": {
        "nest1": "tensor"
    },
    "kwargs": {
        "nest2": "tensor",
        "check_types": "boolean",
        "expand_composites": "boolean"
    },
    "inner": {}
}

signatures["tf.nest.assert_same_structure_2"] = {
    "args": {
        "nest1": "list"
    },
    "kwargs": {
        "nest2": "list",
        "check_types": "boolean",
        "expand_composites": "boolean"
    },
    "inner": {}
}

signatures["tf.nest.assert_same_structure_3"] = {
    "args": {
        "nest1": "tuple"
    },
    "kwargs": {
        "nest2": "tuple",
        "check_types": "boolean",
        "expand_composites": "boolean"
    },
    "inner": {}
}
signatures["tf.nn.crelu"] = {
    "args": {
        "features": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.isotonic_regression"] = {
    "args": {
        "inputs": "tensor"
    },
    "kwargs": {
        "decreasing": "boolean",
        "axis": "integer"
    },
    "inner": {}
}
signatures["tf.nn.log_poisson_loss"] = {
    "args": {
        "targets": "tensor",
        "log_input": "tensor"
    },
    "kwargs": {
        "compute_full_loss": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.softmax"] = {
    "args": {
        "logits": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.random.stateless_gamma_1"] = {
    "args": {
        "shape": "list",
        "seed": "tensor",
        "alpha": "tensor"
    },
    "kwargs": {
        "beta": "tensor",
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}

signatures["tf.random.stateless_gamma_2"] = {
    "args": {
        "shape": "tuple",
        "seed": "tensor",
        "alpha": "tensor"
    },
    "kwargs": {
        "beta": "tensor",
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}

signatures["tf.random.stateless_gamma_3"] = {
    "args": {
        "shape": "tensor",
        "seed": "tensor",
        "alpha": "tensor"
    },
    "kwargs": {
        "beta": "tensor",
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.random.stateless_parameterized_truncated_normal_1"] = {
    "args": {
        "shape": "list"
    },
    "kwargs": {
        "seed": "list",
        "means": "tensor",
        "stddevs": "tensor",
        "minvals": "tensor",
        "maxvals": "tensor",
        "name": "string"
    },
    "inner": {}
}

signatures["tf.random.stateless_parameterized_truncated_normal_2"] = {
    "args": {
        "shape": "list"
    },
    "kwargs": {
        "seed": "list",
        "means": "float",
        "stddevs": "float",
        "minvals": "float",
        "maxvals": "float",
        "name": "string"
    },
    "inner": {}
}

signatures["tf.random.stateless_parameterized_truncated_normal_3"] = {
    "args": {
        "shape": "tensor"
    },
    "kwargs": {
        "seed": "tensor",
        "means": "tensor",
        "stddevs": "tensor",
        "minvals": "tensor",
        "maxvals": "tensor",
        "name": "string"
    },
    "inner": {}
}

signatures["tf.random.stateless_parameterized_truncated_normal_4"] = {
    "args": {
        "shape": "tensor"
    },
    "kwargs": {
        "seed": "tensor",
        "means": "float",
        "stddevs": "float",
        "minvals": "float",
        "maxvals": "float",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.BatchMatMulV3"] = {
    "args": {
        "x": "tensor",
        "y": "tensor",
        "Tout": "dtype"
    },
    "kwargs": {
        "adj_x": "boolean",
        "adj_y": "boolean",
        "grad_x": "boolean",
        "grad_y": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Cosh"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Div"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
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
signatures["tf.raw_ops.Elu"] = {
    "args": {
        "features": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Fact"] = {
    "args": {},
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Gather"] = {
    "args": {
        "params": "tensor",
        "indices": "tensor"
    },
    "kwargs": {
        "validate_indices": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Greater"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.L2Loss"] = {
    "args": {
        "t": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Log"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Maximum"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Mean"] = {
    "args": {
        "input": "tensor",
        "axis": "tensor"
    },
    "kwargs": {
        "keep_dims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Pow"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Prod"] = {
    "args": {
        "input": "tensor",
        "axis": "tensor"
    },
    "kwargs": {
        "keep_dims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.RandomShuffle"] = {
    "args": {
        "value": "tensor"
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
signatures["tf.raw_ops.Round"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Selu"] = {
    "args": {
        "features": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Sinh"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Softplus"] = {
    "args": {
        "features": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.SparseSegmentMean"] = {
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
signatures["tf.raw_ops.Tan"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.UnicodeScript"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Where"] = {
    "args": {
        "condition": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.WriteFile"] = {
    "args": {
        "filename": "string",
        "contents": "string"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.reverse"] = {
    "args": {
        "tensor": "tensor",
        "axis": "list"  # int32/int64 tensor representing indices, but described as 1-D tensor of indices
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.sets.union"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "validate_indices": "boolean"
    },
    "inner": {}
}
signatures["tf.sparse.SparseTensor"] = {
    "args": {
        "indices": "tensor",
        "values": "tensor",
        "dense_shape": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.train.Coordinator"] = {
    "args": {},
    "kwargs": {
        "clean_stop_exception_types": "list"
    },
    "inner": {}
}
signatures["tf.raw_ops.RealDiv"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.LogSoftmax"] = {
    "args": {
        "logits": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.in_top_k"] = {
    "args": {
        "targets": "tensor",
        "predictions": "tensor",
        "k": "integer"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.LinearOperatorLowerTriangular"] = {
    "args": {
        "tril": "tensor"
    },
    "kwargs": {
        "is_non_singular": "boolean",
        "is_self_adjoint": "boolean",
        "is_positive_definite": "boolean",
        "is_square": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.random_jpeg_quality"] = {
    "args": {
        "image": "tensor",
        "min_jpeg_quality": "integer",
        "max_jpeg_quality": "integer"
    },
    "kwargs": {
        "seed": "integer"
    },
    "inner": {}
}
signatures["tf.nn.ctc_beam_search_decoder"] = {
    "args": {
        "inputs": "tensor",
        "sequence_length": "tensor"
    },
    "kwargs": {
        "beam_width": "integer",
        "top_paths": "integer"
    },
    "inner": {}
}
signatures["tf.raw_ops.RandomUniform"] = {
    "args": {
        "shape": "tensor",
    },
    "kwargs": {
        "dtype": "dtype",
        "seed": "integer",
        "seed2": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.FractionalMaxPool"] = {
    "args": {
        "value": "tensor",
        "pooling_ratio": "list"
    },
    "kwargs": {
        "pseudo_random": "boolean",
        "overlapping": "boolean",
        "deterministic": "boolean",
        "seed": "integer",
        "seed2": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.rot90"] = {
    "args": {
        "m": "tensor"
    },
    "kwargs": {
        "k": "integer",
        "axes": "tuple"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.zeros_like"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.nn.ctc_loss_1"] = {
    "args": {
        "labels": "tensor",
        "logits": "tensor",
        "label_length": "tensor",
        "logit_length": "tensor"
    },
    "kwargs": {
        "logits_time_major": "boolean",
        "unique": "tensor",
        "blank_index": "integer",
        "name": "string"
    },
    "inner": {}
}

signatures["tf.nn.ctc_loss_2"] = {
    "args": {
        "labels": "tensor",  # SparseTensor
        "logits": "tensor",
        "label_length": "tensor",  # None for SparseTensor
        "logit_length": "tensor"
    },
    "kwargs": {
        "logits_time_major": "boolean",
        "unique": "tensor",
        "blank_index": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.softsign"] = {
    "args": {
        "features": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.SegmentSum"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Lgamma"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.reciprocal"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.raw_ops.SquaredDifference"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.LRN"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "depth_radius": "integer",
        "bias": "float",
        "alpha": "float",
        "beta": "float",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.nextafter"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.DiagPart"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.floor"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.MatrixSetDiag"] = {
    "args": {
        "input": "tensor",
        "diagonal": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.autodiff.ForwardAccumulator"] = {
    "args": {
        "primals": "tensor",
        "tangents": "tensor"
    },
    "kwargs": {},
    "inner": {
        "args": {},
        "kwargs": {}
    }
}
signatures["tf.nn.compute_accidental_hits"] = {
    "args": {
        "true_classes": "tensor",
        "sampled_candidates": "tensor",
        "num_true": "integer"
    },
    "kwargs": {
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.adjust_gamma"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {
        "gamma": "float",  # scalar or tensor, but typically a number
        "gain": "float"  # scalar or tensor, but typically a number
    },
    "inner": {}
}
signatures["tf.image.stateless_random_hue"] = {
    "args": {
        "image": "tensor",
        "max_delta": "float",
        "seed": "tuple"  # shape [2] Tensor, but can also be tuple of ints
    },
    "kwargs": {},
    "inner": {}
}

signatures["tf.image.stateless_random_hue_2"] = {
    "args": {
        "image": "tensor",
        "max_delta": "float",
        "seed": "tensor"  # Tensor with dtype int32 or int64
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.stateless_random_contrast"] = {
    "args": {
        "image": "tensor",
        "lower": "float",
        "upper": "float",
        "seed": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.compress"] = {
    "args": {
        "condition": "tensor",
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}
signatures["tf.math.unsorted_segment_max"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor",
        "num_segments": "integer"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.bitwise.bitwise_xor"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.sobel_edges"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.raw_ops.EnsureShape_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "shape": "list",
        "name": "string"
    },
    "inner": {}
}

signatures["tf.raw_ops.EnsureShape_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "shape": "tuple",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.flip_up_down"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.strings.unicode_script"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.strided_slice"] = {
    "args": {
        "input_": "tensor",
        "begin": "list",
        "end": "list",
    },
    "kwargs": {
        "strides": "list",
        "begin_mask": "integer",
        "end_mask": "integer",
        "ellipsis_mask": "integer",
        "new_axis_mask": "integer",
        "shrink_axis_mask": "integer",
        "var": "tensor",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.logaddexp"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.sparse.eye"] = {
    "args": {
        "num_rows": "integer"
    },
    "kwargs": {
        "num_columns": "integer",
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.array"] = {
    "args": {
        "val": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "copy": "boolean",
        "ndmin": "integer"
    },
    "inner": {}
}
signatures["tf.linalg.LinearOperatorCirculant2D"] = {
    "args": {
        "spectrum": "tensor"
    },
    "kwargs": {
        "input_output_dtype": "dtype",
        "is_non_singular": "boolean",
        "is_self_adjoint": "boolean",
        "is_positive_definite": "boolean",
        "is_square": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.identity"] = {
    "args": {
        "n": "integer"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.sets.intersection"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "validate_indices": "boolean"
    },
    "inner": {}
}
signatures["tf.image.random_flip_left_right"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {
        "seed": "integer"
    },
    "inner": {}
}
signatures["tf.raw_ops.FloorMod"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.bessel_i0"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Imag"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "Tout": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.expand_dims"] = {
    "args": {
        "a": "tensor",
        "axis": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.special.fresnel_cos"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Erf"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Bucketize"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "boundaries": "list",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Cross"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits"] = {
    "args": {
        "features": "tensor",
        "labels": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.quantization.fake_quant_with_min_max_vars_per_channel_gradient"] = {
    "args": {
        "gradients": "tensor",
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
