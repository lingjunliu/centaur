signatures = {}
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
