signatures = {}
signatures["tf.experimental.numpy.tril"] = {
    "args": {
        "m": "tensor"
    },
    "kwargs": {
        "k": "integer"  # could also be float, but integer seems more appropriate
    },
    "inner": {}
}
signatures["tf.experimental.numpy.floor_divide"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {
    },
    "inner": {}
}
signatures["tf.data.experimental.Counter"] = {
    "args": {
        "start": "integer",  # Could be a tensor, but documentation suggests integer
        "step": "integer"  # Could be a tensor, but documentation suggests integer
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.vdot"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
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
signatures["tf.math.polygamma"] = {
    "args": {
        "a": "tensor",  # Should be float32 or float64, represented as tensor
        "x": "tensor"  # Must have the same type as 'a', represented as tensor
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
signatures["tf.identity_n"] = {
    "args": {
        "input": "tensor_list"  # could also be a single tensor, but the doc says a list
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.diag"] = {
    "args": {
        "v": "tensor"  # Could also be a tensor_list, but documentation states 1- or 2-d. Assuming tensor.
    },
    "kwargs": {
        "k": "integer"  # Could also be float, but documentation implies integer offset.
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
signatures["tf.image.stateless_random_flip_up_down"] = {
    "args": {
        "image": "tensor",  # Could also be tensor_list if it accepts a list of images
        "seed": "tensor"  # specified as shape [2] tensor, int32 or int64
    },
    "kwargs": {
    },
    "inner": {}
}
signatures["tf.raw_ops.Real"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "Tout": "dtype",  # Could be more specific, but dtype seems closest
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
signatures["tf.nest.assert_same_structure"] = {
    "args": {
        "nest1": "tensor",  # could be any nested structure, including atoms
        "nest2": "tensor"  # could be any nested structure, including atoms
    },
    "kwargs": {
        "check_types": "boolean",
        "expand_composites": "boolean"
    },
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
signatures["tf.data.experimental.from_variant"] = {
    "args": {
        "variant": "tensor",  # Should ideally be tf.variant tensor
        "structure": "list"  # Ideally a nested structure of tf.TypeSpec objects. List is the closest we can get.
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.cos"] = {
    "args": {
        "x": "tensor"  # Could also be complex64 or complex128
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
signatures["tf.experimental.numpy.bitwise_not"] = {
    "args": {
        "x": "tensor"  # Could also be tensor_list, but tensor seems more likely
    },
    "kwargs": {},
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
signatures["tf.linalg.lstsq"] = {
    "args": {
        "matrix": "tensor",
        "rhs": "tensor"
    },
    "kwargs": {
        "l2_regularizer": "float",  # Should be a 0-D double Tensor, but float is close enough.
        "fast": "boolean",
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
signatures["tf.math.special.bessel_j1"] = {
    "args": {
        "x": "tensor"  # Could be SparseTensor as well, but tensor is more general
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.reverse"] = {
    "args": {
        "tensor": "tensor",
        "axis": "tensor"  # Could also be int32 or int64, choosing tensor to cover both
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
signatures["tf.convert_to_tensor"] = {
    "args": {
        "value": "tensor" # Could also be list, tuple, or scalar, but "tensor" covers the base case
    },
    "kwargs": {
        "dtype": "dtype",
        "dtype_hint": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.cumsum"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",  # could be None, but integer is the most specific type for an axis
        "dtype": "dtype"
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
signatures["tf.linalg.solve"] = {
    "args": {
        "matrix": "tensor",  # Could also be tensor_list if it accepts a list of matrices
        "rhs": "tensor"
    },
    "kwargs": {
        "adjoint": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.isnan"] = {
    "args": {
        "x": "tensor"  # Could be a tensor_list, but documentation specifies a single 'x'
    },
    "kwargs": {},
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
signatures["tf.compat.forward_compatible"] = {
    "args": {
        "year": "integer",
        "month": "integer",
        "day": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.transpose"] = {
    "args": {
        "image": "tensor"  # Could be either 3D or 4D tensor
    },
    "kwargs": {
        "name": "string"
    },
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
signatures["tf.experimental.numpy.promote_types"] = {
    "args": {
        "type1": "dtype",
        "type2": "dtype"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.stateless_random_flip_left_right"] = {
    "args": {
        "image": "tensor",  # Could also be tensor_list if it accepts a list of images
        "seed": "tensor"  # Expecting a shape [2] tensor, could be int32 or int64
    },
    "kwargs": {},
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
signatures["tf.math.lgamma"] = {
    "args": {
        "x": "tensor"  # Could be more specific about the tensor types (bfloat16, half, float32, float64)
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
signatures["tf.IndexedSlices"] = {
    "args": {
        "values": "tensor",
        "indices": "tensor"  # Should be integer tensor specifically
    },
    "kwargs": {
        "dense_shape": "tensor" # Could be an integer tensor, but document states tensor
    },
    "inner": {}
}
signatures["tf.experimental.numpy.argmin"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer"  # could also be None, but integer is more specific
    },
    "inner": {}
}
signatures["tf.experimental.numpy.append"] = {
    "args": {
        "arr": "tensor",
        "values": "tensor"
    },
    "kwargs": {
        "axis": "integer"  # could also be None, but integer is more specific
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
signatures["tf.raw_ops.Selu"] = {
    "args": {
        "features": "tensor"
    },
    "kwargs": {
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
signatures["tf.image.random_hue"] = {
    "args": {
        "image": "tensor",  # could be tensor_list, but documentation states RGB image, hence tensor
        "max_delta": "float"
    },
    "kwargs": {
        "seed": "integer"  # could be None, but integer is more specific
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
signatures["tf.raw_ops.Round"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.SparseSegmentMean"] = {
    "args": {
        "data": "tensor",
        "indices": "tensor",  # Could be integer as well, but documentation suggests tensor
        "segment_ids": "tensor"  # Could be integer as well, but documentation suggests tensor
    },
    "kwargs": {
        "sparse_gradient": "boolean",
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
        "seed": "integer" # Could be None, but integer is the most restrictive type
    },
    "inner": {}
}
signatures["tf.math.invert_permutation"] = {
    "args": {
        "x": "tensor"  # Could be int32 or int64, choosing tensor as a general type
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.bessel_i1e"] = {
    "args": {
        "x": "tensor"  # Could also be SparseTensor but tensor is more general
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.sparse.SparseTensor"] = {
    "args": {
        "indices": "tensor",  # Should be int64 tensor of shape [N, ndims]
        "values": "tensor",  # Can be any type and shape [N]
        "dense_shape": "tensor"  # Should be int64 tensor of shape [ndims]
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.atan2"] = {
    "args": {
        "y": "tensor",  # Could be bfloat16, half, float32, float64
        "x": "tensor"  # Must have the same type as y
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Gather"] = {
    "args": {
        "params": "tensor",
        "indices": "tensor"  # Could be int32 or int64, using tensor as a general type
    },
    "kwargs": {
        "validate_indices": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.isfinite"] = {
    "args": {
        "x": "tensor"  # Could be tensor_list, but tensor seems more common
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.is_inf"] = {
    "args": {
        "x": "tensor"  # Should be more specific about the tensor types
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
signatures["tf.linalg.LinearOperatorZeros"] = {
    "args": {
        "num_rows": "integer",
        "num_columns": "integer"  # Could be None, but integer is the base type
    },
    "kwargs": {
        "batch_shape": "list",  # Can also be None, but list seems more appropriate
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
signatures["tf.experimental.numpy.isposinf"] = {
    "args": {
        "x": "tensor"  # Could be tensor_list, but documentation specifies single input.
    },
    "kwargs": {
    },
    "inner": {}
}
signatures["tf.random.stateless_parameterized_truncated_normal"] = {
    "args": {
        "shape": "tensor",  # Or list, but tensor seems more appropriate
        "seed": "tensor" # dtype int32 or int64, but "tensor" is the best fit
    },
    "kwargs": {
        "means": "tensor",  # Or float
        "stddevs": "tensor", # Or float
        "minvals": "tensor", # Or float
        "maxvals": "tensor", # Or float
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.erf"] = {
    "args": {
        "x": "tensor"  # Could be bfloat16, half, float32, float64, but tensor is the most general.
    },
    "kwargs": {
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
signatures["tf.linalg.eigh"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.L2Loss"] = {
    "args": {
        "t": "tensor"  # Could be half, bfloat16, float32, or float64, all represented as "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.crelu"] = {
    "args": {
        "features": "tensor"
    },
    "kwargs": {
        "axis": "integer",  # could also be an int32 or int64 tensor
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.floormod"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"  # Should be same type as x
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.isinf"] = {
    "args": {
        "x": "tensor"  # Could be a tensor_list as well, but tensor is more common
    },
    "kwargs": {},
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
signatures["tf.experimental.numpy.ravel"] = {
    "args": {
        "a": "tensor"  # Could potentially be a list of tensors, but documentation suggests a single tensor.
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.resize_with_crop_or_pad"] = {
    "args": {
        "image": "tensor",  # Could also be tensor_list, but doc says 3-D or 4-D tensor
        "target_height": "integer",
        "target_width": "integer"
    },
    "kwargs": {
    },
    "inner": {}
}
signatures["tf.raw_ops.Prod"] = {
    "args": {
        "input": "tensor",
        "axis": "tensor" # Could also be a list of integers, but tensor is the base type
    },
    "kwargs": {
        "keep_dims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.BatchMatMulV3"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "Tout": "dtype",
        "adj_x": "boolean",
        "adj_y": "boolean",
        "grad_x": "boolean",
        "grad_y": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.LinearOperatorHouseholder"] = {
    "args": {
        "reflection_axis": "tensor"  # could be tensor_list if it takes a batch of vectors
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
signatures["tf.experimental.numpy.nanprod"] = {
    "args": {
        "a": "tensor"  # Could be tensor_list, but tensor seems more general
    },
    "kwargs": {
        "axis": "integer", #Could be None, but integer is the most common case
        "dtype": "dtype",
        "keepdims": "boolean"
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
signatures["tf.raw_ops.UnicodeScript"] = {
    "args": {
        "input": "tensor"  # int32 Tensor of Unicode code points
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.dot"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "name": "string"  # Could also be None
    },
    "inner": {}
}
signatures["tf.image.central_crop"] = {
    "args": {
        "image": "tensor",
        "central_fraction": "float"
    },
    "kwargs": {
    },
    "inner": {}
}
signatures["tf.train.Coordinator"] = {
    "args": {},
    "kwargs": {
        "clean_stop_exception_types": "list" # could be a tuple as well
    },
    "inner": {
        "args": {},
        "kwargs": {},
    }
}

signatures["tf.train.Coordinator.stop_on_exception"] = {
    "args": {},
    "kwargs": {},
    "inner": {
        "args": {},
        "kwargs": {},
    }
}

signatures["tf.train.Coordinator.join"] = {
    "args": {
        "threads": "list" # could also be a tuple
    },
    "kwargs": {
        "stop_grace_period_secs": "float" # could be integer
    },
    "inner": {}
}

signatures["tf.train.Coordinator.request_stop"] = {
    "args": {},
    "kwargs": {
        "exception": "string" #could be None, so string
    },
    "inner": {}
}

signatures["tf.train.Coordinator.should_stop"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.fix"] = {
    "args": {
        "x": "tensor"  # Could be a tensor or a number (float/int)
    },
    "kwargs": {
        # "out": "tensor" # Unsupported argument
    },
    "inner": {}
}
signatures["tf.experimental.numpy.moveaxis"] = {
    "args": {
        "a": "tensor",
        "source": "integer", # Could also be a list/tuple of integers
        "destination": "integer" # Could also be a list/tuple of integers
    },
    "kwargs": {
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
signatures["tf.experimental.numpy.conj"] = {
    "args": {
        "x": "tensor"  # Could be tensor_list, but documentation mentions single x.
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.random.stateless_gamma"] = {
    "args": {
        "shape": "tensor",  # Could be a list or tuple of integers, but tensor is more general.
        "seed": "tensor",  # Should be a shape [2] tensor
        "alpha": "tensor",
        "beta": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.sets.union"] = {
    "args": {
        "a": "tensor",  # Could be Tensor or SparseTensor
        "b": "tensor"   # Could be Tensor or SparseTensor
    },
    "kwargs": {
        "validate_indices": "boolean"
    },
    "inner": {}
}
signatures["tf.image.rgb_to_hsv"] = {
    "args": {
        "images": "tensor"  # Could be half, bfloat16, float32, float64
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.logical_not"] = {
    "args": {
        "x": "tensor"  # Should ideally be tensor of type bool
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.broadcast_dynamic_shape"] = {
    "args": {
        "shape_x": "tensor",  # Expected a rank 1 integer tensor
        "shape_y": "tensor"  # Expected a rank 1 integer tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.signbit"] = {
    "args": {
        "x": "tensor"  # Could also be a tensor_list, but tensor seems more general
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.nn.isotonic_regression"] = {
    "args": {
        "inputs": "tensor"  # Could also be tensor_list, but tensor seems more appropriate based on the example
    },
    "kwargs": {
        "decreasing": "boolean",
        "axis": "integer"  # Could also be a tensor of integers, but integer is more general
    },
    "inner": {}
}
signatures["tf.nn.softmax"] = {
    "args": {
        "logits": "tensor"
    },
    "kwargs": {
        "axis": "integer", # Could also be None
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.crop_to_bounding_box"] = {
    "args": {
        "image": "tensor",
        "offset_height": "integer", # Could also be a tensor
        "offset_width": "integer", # Could also be a tensor
        "target_height": "integer", # Could also be a tensor
        "target_width": "integer" # Could also be a tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.raw_ops.Mean"] = {
    "args": {
        "input": "tensor",
        "axis": "tensor"  # Could also be a list of integers
    },
    "kwargs": {
        "keep_dims": "boolean",
        "name": "string"
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
        "logits": "tensor"  # Should be half, bfloat16, float32, or float64 but tensor is most generic
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.in_top_k"] = {
    "args": {
        "targets": "tensor",  # int32 or int64, so tensor is best fit
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
        "tril": "tensor"  # Should it be tensor_list? No, it's a tensor representing a batch of matrices.
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
        "seed": "integer" # Could also be None, but integer covers it
    },
    "inner": {}
}
signatures["tf.nn.ctc_beam_search_decoder"] = {
    "args": {
        "inputs": "tensor",
        "sequence_length": "tensor" # Could also be a list of integers
    },
    "kwargs": {
        "beam_width": "integer",
        "top_paths": "integer"
    },
    "inner": {}
}
signatures["tf.raw_ops.RandomUniform"] = {
    "args": {
        "shape": "tensor",  # Can be int32 or int64, using tensor as a general type
        "dtype": "dtype"
    },
    "kwargs": {
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
        "k": "integer", # Could also be float, but integer is more precise based on numpy docs
        "axes": "tuple"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.zeros_like"] = {
    "args": {
        "a": "tensor"  # Could be a list of tensors, but 'tensor' seems more likely as the primary use case
    },
    "kwargs": {
        "dtype": "dtype"  # Could be None, but dtype is the appropriate type.
    },
    "inner": {}
}
signatures["tf.nn.ctc_loss"] = {
    "args": {
        "labels": "tensor",  # Could also be SparseTensor
        "logits": "tensor",
        "label_length": "tensor_list",
        "logit_length": "tensor_list"
    },
    "kwargs": {
        "logits_time_major": "boolean",
        "unique": "tensor", # could be None
        "blank_index": "integer",  # Could be None
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.softsign"] = {
    "args": {
        "features": "tensor"  # Could be half, bfloat16, float32, float64 - all tensors
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
        "x": "tensor"  # Can be bfloat16, half, float32, float64
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.reciprocal"] = {
    "args": {
        "x": "tensor"  # Could also be a tensor_list, but tensor seems more general
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
        "x1": "tensor",  # Can be float64 or float32, so tensor is the best match.
        "x2": "tensor"  # Must have the same type as x1, so tensor.
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
        "x": "tensor"  # could also be tensor_list, but documentation mentions a single Tensor.
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
        "args": {
            "x": "tensor"  # Could also be a list of tensors
        },
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
        "gamma": "float",  # Can also be a tensor
        "gain": "float"  # Can also be a tensor
    },
    "inner": {}
}
signatures["tf.image.stateless_random_hue"] = {
    "args": {
        "image": "tensor",  # could also be tensor_list
        "max_delta": "float",
        "seed": "tensor"  # Shape [2] Tensor, dtype int32 or int64, so tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.stateless_random_contrast"] = {
    "args": {
        "image": "tensor",
        "lower": "float",
        "upper": "float",
        "seed": "tensor"  # Should be a shape [2] Tensor
    },
    "kwargs": {
    },
    "inner": {}
}
signatures["tf.experimental.numpy.compress"] = {
    "args": {
        "condition": "tensor",  # could also be a list of booleans
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer"  # could be None
    },
    "inner": {}
}
signatures["tf.math.unsorted_segment_max"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor",
        "num_segments": "integer" # Could also be tensor, but integer seems more precise.
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.bitwise.bitwise_xor"] = {
    "args": {
        "x": "tensor",  # Can be int8, int16, int32, int64, uint8, uint16, uint32, uint64
        "y": "tensor"  # Must have the same type as x
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.sobel_edges"] = {
    "args": {
        "image": "tensor"  # could also be tensor_list, but documentation mentions shape [batch_size, h, w, d], hence tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.raw_ops.EnsureShape"] = {
    "args": {
        "input": "tensor",
        "shape": "list" # Could also be tensor, but list seems more accurate given the documentation
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.flip_up_down"] = {
    "args": {
        "image": "tensor"  # Could also be tensor_list, but the example uses a tensor.
    },
    "kwargs": {
    },
    "inner": {}
}
signatures["tf.strings.unicode_script"] = {
    "args": {
        "input": "tensor"  # Accepts a Tensor of int32 Unicode code points
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.strided_slice"] = {
    "args": {
        "input_": "tensor",
        "begin": "tensor_list",  # could be int32 or int64 tensor
        "end": "tensor_list",  # could be int32 or int64 tensor
        "strides": "tensor_list"  # could be int32 or int64 tensor
    },
    "kwargs": {
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
        "num_rows": "integer",  # could also be tensor
        "num_columns": "integer" # could also be tensor
    },
    "kwargs": {
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.array"] = {
    "args": {
        "val": "tensor"  # Can also be ndarray, but tensor is the most representative type here
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
        "input_output_dtype": "dtype", # Could also be tensor
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
        "n": "integer"  # Could be tensor of integers, but documentation specifies integer
    },
    "kwargs": {
        "dtype": "dtype"  # Should be tf.DType
    },
    "inner": {}
}
signatures["tf.sets.intersection"] = {
    "args": {
        "a": "tensor",  # Could also be SparseTensor, but using tensor as a general type
        "b": "tensor"  # Could also be SparseTensor, but using tensor as a general type
    },
    "kwargs": {
        "validate_indices": "boolean"
    },
    "inner": {}
}
signatures["tf.image.random_flip_left_right"] = {
    "args": {
        "image": "tensor"  # Can be 3D or 4D tensor
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
        "x": "tensor"  # Could also be SparseTensor but tensor is more general
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
        "Tout": "dtype", # Could be tensor, but dtype is more appropriate since it specifies the output type
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.expand_dims"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer"  # Could also be a tuple of integers
    },
    "inner": {}
}
signatures["tf.math.special.fresnel_cos"] = {
    "args": {
        "x": "tensor"  # Could also be SparseTensor
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
        "input": "tensor",  # Can be int32, int64, float32, float64
        "boundaries": "list"  # List of floats
    },
    "kwargs": {
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
        "features": "tensor",  # could be half, bfloat16, float32, float64
        "labels": "tensor"  # could be int32, int64
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
        "num_bits": "integer",  # Could be int32 or int64, choosing int
        "narrow_range": "boolean",
        "name": "string"
    },
    "inner": {}
}
