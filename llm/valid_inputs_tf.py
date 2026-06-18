generated_inputs = {}

import tensorflow as tf
import numpy as np
import copy

def tf_bitcast_inputs():
    list_of_inputs = []
    
    # 1. float32 to uint8 (larger to smaller)
    list_of_inputs.append({
        'input': np.array([1.0, -2.0, 3.5], dtype=np.float32),
        'type': np.dtype('uint8'),
        'name': 'bitcast_1'
    })
    
    # 2. uint8 to float32 (smaller to larger)
    list_of_inputs.append({
        'input': np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint8),
        'type': np.dtype('float32'),
        'name': 'bitcast_2'
    })
    
    # 3. int32 to float32 (equal size)
    list_of_inputs.append({
        'input': np.array([-1, 0, 5], dtype=np.int32),
        'type': np.dtype('float32'),
        'name': 'bitcast_3'
    })
    
    # 4. float64 to int64 (equal size)
    list_of_inputs.append({
        'input': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        'type': np.dtype('int64'),
        'name': 'bitcast_4'
    })
    
    # 5. int16 to uint8 (larger to smaller)
    list_of_inputs.append({
        'input': np.array([10, -10], dtype=np.int16),
        'type': np.dtype('uint8'),
        'name': 'bitcast_5'
    })
    
    # 6. uint8 to int16 (smaller to larger)
    list_of_inputs.append({
        'input': np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8),
        'type': np.dtype('int16'),
        'name': 'bitcast_6'
    })
    
    # 7. float32 to int32 (equal size, 3D)
    list_of_inputs.append({
        'input': np.array([[[1.5]]], dtype=np.float32),
        'type': np.dtype('int32'),
        'name': 'bitcast_7'
    })
    
    # 8. complex64 to float32 (larger to smaller)
    list_of_inputs.append({
        'input': np.array([1.0 + 2.0j], dtype=np.complex64),
        'type': np.dtype('float32'),
        'name': 'bitcast_8'
    })
    
    # 9. float32 to complex64 (smaller to larger)
    list_of_inputs.append({
        'input': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        'type': np.dtype('complex64'),
        'name': 'bitcast_9'
    })
    
    # 10. int64 to uint8 (larger to smaller)
    list_of_inputs.append({
        'input': np.array([123456789], dtype=np.int64),
        'type': np.dtype('uint8'),
        'name': 'bitcast_10'
    })
    
    return list_of_inputs

generated_inputs["tf.bitcast"] = tf_bitcast_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_bitwise_bitwise_xor_inputs():
    list_of_inputs = []
    
    # Input 1: 1D arrays, int32
    list_of_inputs.append({
        "x": np.array([0, 5, 3, 14], dtype=np.int32),
        "y": np.array([5, 0, 7, 11], dtype=np.int32),
        "name": "xor_int32"
    })
    
    # Input 2: 2D arrays, int32 with negative values
    list_of_inputs.append({
        "x": np.array([[-1, 2], [-3, 4]], dtype=np.int32),
        "y": np.array([[5, -6], [7, -8]], dtype=np.int32),
        "name": "xor_int32_negative"
    })

    # Input 3: 1D arrays, int32 positive
    list_of_inputs.append({
        "x": np.array([255, 0, 127], dtype=np.int32),
        "y": np.array([0, 255, 128], dtype=np.int32),
        "name": "xor_int32_positive"
    })

    # Input 4: Scalar values (0D arrays), int64
    list_of_inputs.append({
        "x": np.array(922337203685477580, dtype=np.int64),
        "y": np.array(-922337203685477580, dtype=np.int64),
        "name": "xor_scalar_int64"
    })

    # Input 5: 3D arrays, int32
    list_of_inputs.append({
        "x": np.array([[[1, -2], [3, -4]], [[5, -6], [7, -8]]], dtype=np.int32),
        "y": np.array([[[-1, 2], [-3, 4]], [[-5, 6], [-7, 8]]], dtype=np.int32),
        "name": "xor_int32_3d"
    })

    # Input 6: 4D arrays, int32
    list_of_inputs.append({
        "x": np.ones((2, 2, 2, 2), dtype=np.int32) * 65535,
        "y": np.zeros((2, 2, 2, 2), dtype=np.int32),
        "name": "xor_int32_4d"
    })

    # Input 7: 2D arrays, int64
    list_of_inputs.append({
        "x": np.array([[123456, 789012], [345678, 901234]], dtype=np.int64),
        "y": np.array([[654321, 210987], [876543, 432109]], dtype=np.int64),
        "name": "xor_int64_2d"
    })

    # Input 8: Broadcasting 2D and 1D arrays, int32
    list_of_inputs.append({
        "x": np.array([[1], [2], [3]], dtype=np.int32),
        "y": np.array([4, 5, 6], dtype=np.int32),
        "name": "xor_broadcast"
    })

    # Input 9: Large values in int64
    list_of_inputs.append({
        "x": np.array([4611686018427387903, 0], dtype=np.int64),
        "y": np.array([0, 4611686018427387903], dtype=np.int64),
        "name": "xor_int64_large"
    })

    # Input 10: 1D arrays of int64 with negative values
    list_of_inputs.append({
        "x": np.array([-100, 200, -300], dtype=np.int64),
        "y": np.array([400, -500, 600], dtype=np.int64),
        "name": "xor_int64_1d"
    })

    return list_of_inputs

generated_inputs["tf.bitwise.bitwise_xor"] = tf_bitwise_bitwise_xor_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_dtypes_saturate_cast_inputs():
    list_of_inputs = []

    # Input 1
    list_of_inputs.append({
        "value": np.array([1e10, -1e10, 5.5], dtype=np.float32),
        "dtype": np.int32,
        "name": "cast_1"
    })

    # Input 2
    list_of_inputs.append({
        "value": np.array([[-10.5, 256.1], [100.0, -1.0]], dtype=np.float32),
        "dtype": np.uint8,
        "name": "cast_2"
    })

    # Input 3
    list_of_inputs.append({
        "value": np.array(40000.0, dtype=np.float64),
        "dtype": np.int16,
        "name": "cast_3"
    })

    # Input 4
    list_of_inputs.append({
        "value": np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64),
        "dtype": np.float32,
        "name": "cast_4"
    })

    # Input 5
    list_of_inputs.append({
        "value": np.array([150, -200, 50, -50], dtype=np.int32),
        "dtype": np.int8,
        "name": "cast_5"
    })

    # Input 6
    list_of_inputs.append({
        "value": np.array([[[[1000.0], [2000.0]], [[3000.0], [4000.0]]]], dtype=np.float32),
        "dtype": np.float16,
        "name": "cast_6"
    })

    # Input 7
    list_of_inputs.append({
        "value": np.array([[70000, -5], [100, 500]], dtype=np.int32),
        "dtype": np.uint16,
        "name": "cast_7"
    })

    # Input 8
    list_of_inputs.append({
        "value": np.array([np.inf, -np.inf, 1.5], dtype=np.float64),
        "dtype": np.float32,
        "name": "cast_8"
    })

    # Input 9
    list_of_inputs.append({
        "value": np.array([[[128, -129], [0, 1]], [[127, -128], [5, -5]]], dtype=np.int32),
        "dtype": np.int8,
        "name": "cast_9"
    })

    # Input 10
    list_of_inputs.append({
        "value": np.array([[5e9, -100], [10, 20]], dtype=np.float32),
        "dtype": np.uint32,
        "name": "cast_10"
    })

    return list_of_inputs

generated_inputs["tf.dtypes.saturate_cast"] = tf_dtypes_saturate_cast_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_histogram_fixed_width_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'values': np.array([-1.0, 0.0, 1.5, 2.0, 5.0, 15], dtype=np.float32),
        'value_range': np.array([0.0, 5.0], dtype=np.float32),
        'nbins': 5,
        'dtype': np.int32,
        'name': 'hist_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'values': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        'value_range': np.array([0.0, 10.0], dtype=np.float64),
        'nbins': 10,
        'dtype': np.int64,
        'name': 'hist_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'values': np.array([-10, -5, 0, 5, 10], dtype=np.int32),
        'value_range': np.array([-15, 15], dtype=np.int32),
        'nbins': 3,
        'dtype': np.int32,
        'name': 'hist_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'values': np.random.uniform(-1.0, 1.0, size=(10, 10)).astype(np.float32),
        'value_range': np.array([-1.0, 1.0], dtype=np.float32),
        'nbins': 20,
        'dtype': np.int32,
        'name': 'hist_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'values': np.random.normal(0, 1, size=(5, 5, 5)).astype(np.float32),
        'value_range': np.array([-3.0, 3.0], dtype=np.float32),
        'nbins': 15,
        'dtype': np.int64,
        'name': 'hist_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'values': np.array([100.5], dtype=np.float32),
        'value_range': np.array([100.0, 101.0], dtype=np.float32),
        'nbins': 2,
        'dtype': np.int32,
        'name': 'hist_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'values': np.arange(10, dtype=np.float64),
        'value_range': np.array([0.0, 9.0], dtype=np.float64),
        'nbins': 9,
        'dtype': np.int32,
        'name': 'hist_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'values': np.array([[[[1, 2], [3, 4]]]], dtype=np.int32),
        'value_range': np.array([1, 4], dtype=np.int32),
        'nbins': 4,
        'dtype': np.int64,
        'name': 'hist_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'values': np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6], dtype=np.float32),
        'value_range': np.array([1.0, 2.0], dtype=np.float32),
        'nbins': 10,
        'dtype': np.int32,
        'name': 'hist_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'values': np.linspace(-100, 100, 50, dtype=np.float32),
        'value_range': np.array([-50.0, 50.0], dtype=np.float32),
        'nbins': 5,
        'dtype': np.int32,
        'name': 'hist_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.histogram_fixed_width"] = tf_histogram_fixed_width_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_histogram_fixed_width_bins_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 input with 5 bins
    input_dict = {
        'values': np.array([-1.0, 0.0, 1.5, 2.0, 5.0, 15.0], dtype=np.float32),
        'value_range': np.array([0.0, 5.0], dtype=np.float32),
        'nbins': 5,
        'dtype': np.int32,
        'name': 'bins_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 input, 10 bins, int64 output dtype
    input_dict = {
        'values': np.array([[-10.0, -5.0], [0.0, 5.0]], dtype=np.float64),
        'value_range': np.array([-10.0, 10.0], dtype=np.float64),
        'nbins': 10,
        'dtype': np.int64,
        'name': 'bins_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float32 values, 5 bins, negative and positive
    input_dict = {
        'values': np.array([[[1.0, 2.0], [3.0, -4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32),
        'value_range': np.array([-5.0, 10.0], dtype=np.float32),
        'nbins': 5,
        'dtype': np.int32,
        'name': 'bins_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Single element float32 array, many bins
    input_dict = {
        'values': np.array([0.55], dtype=np.float32),
        'value_range': np.array([0.0, 1.0], dtype=np.float32),
        'nbins': 100,
        'dtype': np.int32,
        'name': 'bins_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float32 values with negative range bounds
    input_dict = {
        'values': np.array([-3.5, -2.1, -1.0, -0.5], dtype=np.float32),
        'value_range': np.array([-5.0, -1.0], dtype=np.float32),
        'nbins': 4,
        'dtype': np.int32,
        'name': 'bins_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Scalar/0D float32 value
    input_dict = {
        'values': np.array(2.5, dtype=np.float32),
        'value_range': np.array([0.0, 5.0], dtype=np.float32),
        'nbins': 5,
        'dtype': np.int32,
        'name': 'bins_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float64 values with very large range and bin size
    input_dict = {
        'values': np.array([1e5, 2e5, 3e5], dtype=np.float64),
        'value_range': np.array([0.0, 1e6], dtype=np.float64),
        'nbins': 1000,
        'dtype': np.int32,
        'name': 'bins_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float64 values, different bounds
    input_dict = {
        'values': np.array([-1.5, 0.5, 1.5], dtype=np.float64),
        'value_range': np.array([-2.0, 2.0], dtype=np.float64),
        'nbins': 4,
        'dtype': np.int32,
        'name': 'bins_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: High-dimensional float32 array using generated random values
    input_dict = {
        'values': np.random.uniform(-10, 10, (2, 2, 2, 2)).astype(np.float32),
        'value_range': np.array([-10.0, 10.0], dtype=np.float32),
        'nbins': 10,
        'dtype': np.int32,
        'name': 'bins_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float64 values and Int64 output dtype
    input_dict = {
        'values': np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64),
        'value_range': np.array([0.0, 100.0], dtype=np.float64),
        'nbins': 10,
        'dtype': np.int64,
        'name': 'bins_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.histogram_fixed_width_bins"] = tf_histogram_fixed_width_bins_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_combined_non_max_suppression_inputs():
    list_of_inputs = []
    
    np.random.seed(42)
    
    def get_valid_boxes(shape):
        boxes = np.random.rand(*shape).astype(np.float32)
        y1 = np.minimum(boxes[..., 0], boxes[..., 2])
        y2 = np.maximum(boxes[..., 0], boxes[..., 2])
        x1 = np.minimum(boxes[..., 1], boxes[..., 3])
        x2 = np.maximum(boxes[..., 1], boxes[..., 3])
        boxes[..., 0] = y1
        boxes[..., 1] = x1
        boxes[..., 2] = y2
        boxes[..., 3] = x2
        return boxes

    # Input 1
    boxes = get_valid_boxes([2, 5, 1, 4])
    scores = np.random.rand(2, 5, 3).astype(np.float32)
    list_of_inputs.append({
        'boxes': boxes,
        'scores': scores,
        'max_output_size_per_class': 2,
        'max_total_size': 5,
        'iou_threshold': 0.5,
        'score_threshold': 0.1,
        'pad_per_class': False,
        'clip_boxes': True,
        'name': 'nms_1'
    })

    # Input 2
    boxes = get_valid_boxes([1, 10, 2, 4])
    scores = np.random.rand(1, 10, 2).astype(np.float32)
    list_of_inputs.append({
        'boxes': boxes,
        'scores': scores,
        'max_output_size_per_class': 3,
        'max_total_size': 4,
        'iou_threshold': 0.6,
        'score_threshold': 0.0,
        'pad_per_class': True,
        'clip_boxes': False,
        'name': 'nms_2'
    })

    # Input 3
    boxes = get_valid_boxes([3, 4, 1, 4])
    scores = np.random.rand(3, 4, 1).astype(np.float32)
    list_of_inputs.append({
        'boxes': boxes,
        'scores': scores,
        'max_output_size_per_class': 1,
        'max_total_size': 2,
        'iou_threshold': 0.3,
        'score_threshold': 0.5,
        'pad_per_class': False,
        'clip_boxes': True,
        'name': 'nms_3'
    })

    # Input 4
    boxes = get_valid_boxes([2, 8, 4, 4])
    scores = np.random.rand(2, 8, 4).astype(np.float32) - 0.5
    list_of_inputs.append({
        'boxes': boxes,
        'scores': scores,
        'max_output_size_per_class': 4,
        'max_total_size': 10,
        'iou_threshold': 0.7,
        'score_threshold': -0.2,
        'pad_per_class': True,
        'clip_boxes': True,
        'name': 'nms_4'
    })

    # Input 5
    boxes = get_valid_boxes([1, 1, 1, 4])
    scores = np.random.rand(1, 1, 1).astype(np.float32)
    list_of_inputs.append({
        'boxes': boxes,
        'scores': scores,
        'max_output_size_per_class': 1,
        'max_total_size': 1,
        'iou_threshold': 0.5,
        'score_threshold': 0.0,
        'pad_per_class': False,
        'clip_boxes': True,
        'name': 'nms_5'
    })

    # Input 6
    boxes = get_valid_boxes([4, 15, 1, 4])
    scores = np.random.rand(4, 15, 2).astype(np.float32)
    list_of_inputs.append({
        'boxes': boxes,
        'scores': scores,
        'max_output_size_per_class': 5,
        'max_total_size': 10,
        'iou_threshold': 0.4,
        'score_threshold': 0.2,
        'pad_per_class': False,
        'clip_boxes': False,
        'name': 'nms_6'
    })

    # Input 7
    boxes = get_valid_boxes([2, 2, 2, 4])
    scores = np.random.rand(2, 2, 2).astype(np.float32)
    list_of_inputs.append({
        'boxes': boxes,
        'scores': scores,
        'max_output_size_per_class': 2,
        'max_total_size': 4,
        'iou_threshold': 0.1,
        'score_threshold': 0.9,
        'pad_per_class': True,
        'clip_boxes': True,
        'name': 'nms_7'
    })

    # Input 8
    boxes = get_valid_boxes([1, 5, 1, 4])
    scores = (np.random.rand(1, 5, 3).astype(np.float32) - 1.5)
    list_of_inputs.append({
        'boxes': boxes,
        'scores': scores,
        'max_output_size_per_class': 2,
        'max_total_size': 6,
        'iou_threshold': 0.9,
        'score_threshold': -1.0,
        'pad_per_class': False,
        'clip_boxes': False,
        'name': 'nms_8'
    })

    # Input 9
    boxes = get_valid_boxes([5, 12, 1, 4])
    scores = np.random.rand(5, 12, 1).astype(np.float32)
    list_of_inputs.append({
        'boxes': boxes,
        'scores': scores,
        'max_output_size_per_class': 3,
        'max_total_size': 15,
        'iou_threshold': 0.5,
        'score_threshold': 0.05,
        'pad_per_class': True,
        'clip_boxes': True,
        'name': 'nms_9'
    })

    # Input 10
    boxes = get_valid_boxes([2, 6, 3, 4])
    scores = np.random.rand(2, 6, 3).astype(np.float32)
    list_of_inputs.append({
        'boxes': boxes,
        'scores': scores,
        'max_output_size_per_class': 1,
        'max_total_size': 3,
        'iou_threshold': 0.8,
        'score_threshold': 0.3,
        'pad_per_class': False,
        'clip_boxes': False,
        'name': 'nms_10'
    })

    return list_of_inputs

generated_inputs["tf.image.combined_non_max_suppression"] = tf_image_combined_non_max_suppression_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_non_max_suppression_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case with 2 boxes
    list_of_inputs.append({
        'boxes': np.array([[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 1.1, 1.1]], dtype=np.float32),
        'scores': np.array([0.9, 0.8], dtype=np.float32),
        'max_output_size': int(1),
        'iou_threshold': float(0.5),
        'score_threshold': float(0.0),
        'name': 'nms_1'
    })

    # Input 2: Multiple non-overlapping boxes
    list_of_inputs.append({
        'boxes': np.array([[float(i), float(i), float(i + 1), float(i + 1)] for i in range(10)], dtype=np.float32),
        'scores': np.array([float(i) / 10.0 for i in range(10)], dtype=np.float32),
        'max_output_size': int(5),
        'iou_threshold': float(0.5),
        'score_threshold': float(0.1),
        'name': 'nms_2'
    })

    # Input 3: Heavy overlap, low max_output_size
    list_of_inputs.append({
        'boxes': np.array([[0.0, 0.0, 1.0, 1.0]] * 5, dtype=np.float32),
        'scores': np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        'max_output_size': int(2),
        'iou_threshold': float(0.1),
        'score_threshold': float(0.0),
        'name': 'nms_3'
    })

    # Input 4: Negative coordinates and negative scores
    list_of_inputs.append({
        'boxes': np.array([[-10.0, -10.0, 0.0, 0.0], [-9.0, -9.0, 1.0, 1.0]], dtype=np.float32),
        'scores': np.array([-10.0, -5.0], dtype=np.float32),
        'max_output_size': int(2),
        'iou_threshold': float(0.8),
        'score_threshold': float(-15.0),
        'name': 'nms_4'
    })

    # Input 5: Large scale coordinates, high iou_threshold
    list_of_inputs.append({
        'boxes': np.array([[100.0, 100.0, 200.0, 200.0], [120.0, 120.0, 220.0, 220.0], [300.0, 300.0, 400.0, 400.0]], dtype=np.float32),
        'scores': np.array([0.9, 0.75, 0.6], dtype=np.float32),
        'max_output_size': int(3),
        'iou_threshold': float(0.9),
        'score_threshold': float(0.5),
        'name': 'nms_5'
    })

    # Input 6: Zero boxes (empty case)
    list_of_inputs.append({
        'boxes': np.empty((0, 4), dtype=np.float32),
        'scores': np.empty((0,), dtype=np.float32),
        'max_output_size': int(5),
        'iou_threshold': float(0.5),
        'score_threshold': float(0.0),
        'name': 'nms_6'
    })

    # Input 7: Only one box
    list_of_inputs.append({
        'boxes': np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
        'scores': np.array([0.5], dtype=np.float32),
        'max_output_size': int(10),
        'iou_threshold': float(0.5),
        'score_threshold': float(0.1),
        'name': 'nms_7'
    })

    # Input 8: High score_threshold (all pruned)
    list_of_inputs.append({
        'boxes': np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]], dtype=np.float32),
        'scores': np.array([0.3, 0.4], dtype=np.float32),
        'max_output_size': int(2),
        'iou_threshold': float(0.5),
        'score_threshold': float(0.5),
        'name': 'nms_8'
    })

    # Input 9: iou_threshold = 0.0 (aggressive suppression)
    list_of_inputs.append({
        'boxes': np.array([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]], dtype=np.float32),
        'scores': np.array([0.9, 0.8], dtype=np.float32),
        'max_output_size': int(2),
        'iou_threshold': float(0.0),
        'score_threshold': float(0.1),
        'name': 'nms_9'
    })

    # Input 10: iou_threshold = 1.0 (no suppression for overlaps)
    list_of_inputs.append({
        'boxes': np.array([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]], dtype=np.float32),
        'scores': np.array([0.9, 0.8], dtype=np.float32),
        'max_output_size': int(2),
        'iou_threshold': float(1.0),
        'score_threshold': float(0.1),
        'name': 'nms_10'
    })

    return list_of_inputs

generated_inputs["tf.image.non_max_suppression"] = tf_image_non_max_suppression_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_non_max_suppression_overlaps_inputs():
    list_of_inputs = []
    
    # Input 1
    overlaps1 = np.array([[1.0, 0.4, 0.3], [0.4, 1.0, 0.2], [0.3, 0.2, 1.0]], dtype=np.float32)
    scores1 = np.array([0.9, 0.75, 0.6], dtype=np.float32)
    input_dict1 = {
        'overlaps': overlaps1,
        'scores': scores1,
        'max_output_size': int(2),
        'overlap_threshold': float(0.5),
        'score_threshold': float(0.1),
        'name': "nms_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    overlaps2 = np.array([[1.0]], dtype=np.float32)
    scores2 = np.array([0.5], dtype=np.float32)
    input_dict2 = {
        'overlaps': overlaps2,
        'scores': scores2,
        'max_output_size': int(1),
        'overlap_threshold': float(0.3),
        'score_threshold': float(0.0),
        'name': "nms_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    overlaps3 = np.eye(10, dtype=np.float32)
    scores3 = np.linspace(0.1, 1.0, 10, dtype=np.float32)
    input_dict3 = {
        'overlaps': overlaps3,
        'scores': scores3,
        'max_output_size': int(5),
        'overlap_threshold': float(0.5),
        'score_threshold': float(0.5),
        'name': "nms_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    overlaps4 = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.4], [0.2, 0.4, 1.0]], dtype=np.float32)
    scores4 = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    input_dict4 = {
        'overlaps': overlaps4,
        'scores': scores4,
        'max_output_size': int(2),
        'overlap_threshold': float(0.7),
        'score_threshold': float(0.0),
        'name': "nms_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    overlaps5 = np.eye(4, dtype=np.float32)
    scores5 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    input_dict5 = {
        'overlaps': overlaps5,
        'scores': scores5,
        'max_output_size': int(10),
        'overlap_threshold': float(0.1),
        'score_threshold': float(-1.0),
        'name': "nms_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6
    overlaps6 = np.ones((6, 6), dtype=np.float32)
    scores6 = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float32)
    input_dict6 = {
        'overlaps': overlaps6,
        'scores': scores6,
        'max_output_size': int(1),
        'overlap_threshold': float(0.99),
        'score_threshold': float(0.8),
        'name': "nms_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7
    temp = np.arange(16, dtype=np.float32).reshape(4, 4)
    overlaps7 = (temp + temp.T) / 32.0
    np.fill_diagonal(overlaps7, 1.0)
    scores7 = np.array([0.9, 0.1, 0.2, 0.8], dtype=np.float32)
    input_dict7 = {
        'overlaps': overlaps7,
        'scores': scores7,
        'max_output_size': int(3),
        'overlap_threshold': float(0.45),
        'score_threshold': float(0.2),
        'name': "nms_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8
    overlaps8 = np.eye(2, dtype=np.float32)
    scores8 = np.array([-0.1, -0.2], dtype=np.float32)
    input_dict8 = {
        'overlaps': overlaps8,
        'scores': scores8,
        'max_output_size': int(2),
        'overlap_threshold': float(0.5),
        'score_threshold': float(-0.5),
        'name': "nms_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9
    overlaps9 = np.eye(5, dtype=np.float32) * 0.9
    scores9 = np.array([0.95, 0.85, 0.75, 0.65, 0.55], dtype=np.float32)
    input_dict9 = {
        'overlaps': overlaps9,
        'scores': scores9,
        'max_output_size': int(0),
        'overlap_threshold': float(0.5),
        'score_threshold': float(0.0),
        'name': "nms_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10
    overlaps10 = np.array([[1.0, 0.9, 0.0, 0.0], [0.9, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.9], [0.0, 0.0, 0.9, 1.0]], dtype=np.float32)
    scores10 = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    input_dict10 = {
        'overlaps': overlaps10,
        'scores': scores10,
        'max_output_size': int(2),
        'overlap_threshold': float(0.8),
        'score_threshold': float(0.5),
        'name': "nms_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.image.non_max_suppression_overlaps"] = tf_image_non_max_suppression_overlaps_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_image_non_max_suppression_with_scores_inputs():
    list_of_inputs = []

    # Case 1: Standard NMS, basic boxes
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1], [0.5, 0.5, 1.5, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.75, 0.6], dtype=np.float32)
    max_output_size = 2
    iou_threshold = 0.5
    score_threshold = 0.0
    soft_nms_sigma = 0.0
    name = "nms_standard"
    list_of_inputs.append({
        'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size,
        'iou_threshold': iou_threshold, 'score_threshold': score_threshold,
        'soft_nms_sigma': soft_nms_sigma, 'name': name
    })

    # Case 2: Soft NMS enabled
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1], [0.0, 0.2, 1.0, 1.2]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = 3
    iou_threshold = 0.5
    score_threshold = 0.1
    soft_nms_sigma = 0.5
    name = "nms_soft"
    list_of_inputs.append({
        'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size,
        'iou_threshold': iou_threshold, 'score_threshold': score_threshold,
        'soft_nms_sigma': soft_nms_sigma, 'name': name
    })

    # Case 3: Empty boxes (0 boxes)
    boxes = np.empty((0, 4), dtype=np.float32)
    scores = np.empty((0,), dtype=np.float32)
    max_output_size = 5
    iou_threshold = 0.5
    score_threshold = 0.0
    soft_nms_sigma = 0.0
    name = "nms_empty"
    list_of_inputs.append({
        'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size,
        'iou_threshold': iou_threshold, 'score_threshold': score_threshold,
        'soft_nms_sigma': soft_nms_sigma, 'name': name
    })

    # Case 4: Negative scores
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]], dtype=np.float32)
    scores = np.array([-0.5, -0.1], dtype=np.float32)
    max_output_size = 2
    iou_threshold = 0.5
    score_threshold = -1.0
    soft_nms_sigma = 0.0
    name = "nms_neg_scores"
    list_of_inputs.append({
        'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size,
        'iou_threshold': iou_threshold, 'score_threshold': score_threshold,
        'soft_nms_sigma': soft_nms_sigma, 'name': name
    })

    # Case 5: Standard NMS, very high IoU threshold
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.01, 1.0, 1.01]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    max_output_size = 2
    iou_threshold = 0.99
    score_threshold = 0.0
    soft_nms_sigma = 0.0
    name = "nms_high_iou"
    list_of_inputs.append({
        'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size,
        'iou_threshold': iou_threshold, 'score_threshold': score_threshold,
        'soft_nms_sigma': soft_nms_sigma, 'name': name
    })

    # Case 6: Standard NMS, very low IoU threshold
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.5, 1.0, 1.5]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    max_output_size = 2
    iou_threshold = 0.01
    score_threshold = 0.0
    soft_nms_sigma = 0.0
    name = "nms_low_iou"
    list_of_inputs.append({
        'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size,
        'iou_threshold': iou_threshold, 'score_threshold': score_threshold,
        'soft_nms_sigma': soft_nms_sigma, 'name': name
    })

    # Case 7: High max_output_size (larger than num_boxes)
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]], dtype=np.float32)
    scores = np.array([0.5, 0.6], dtype=np.float32)
    max_output_size = 100
    iou_threshold = 0.5
    score_threshold = 0.0
    soft_nms_sigma = 0.0
    name = "nms_large_max_out"
    list_of_inputs.append({
        'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size,
        'iou_threshold': iou_threshold, 'score_threshold': score_threshold,
        'soft_nms_sigma': soft_nms_sigma, 'name': name
    })

    # Case 8: Small max_output_size (1 box allowed)
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0], [4.0, 4.0, 5.0, 5.0]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    max_output_size = 1
    iou_threshold = 0.5
    score_threshold = 0.0
    soft_nms_sigma = 0.0
    name = "nms_small_max_out"
    list_of_inputs.append({
        'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size,
        'iou_threshold': iou_threshold, 'score_threshold': score_threshold,
        'soft_nms_sigma': soft_nms_sigma, 'name': name
    })

    # Case 9: All boxes identical
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    scores = np.array([0.9, 0.9, 0.9], dtype=np.float32)
    max_output_size = 3
    iou_threshold = 0.5
    score_threshold = 0.0
    soft_nms_sigma = 0.0
    name = "nms_identical"
    list_of_inputs.append({
        'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size,
        'iou_threshold': iou_threshold, 'score_threshold': score_threshold,
        'soft_nms_sigma': soft_nms_sigma, 'name': name
    })

    # Case 10: Soft NMS with high sigma
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    max_output_size = 2
    iou_threshold = 0.5
    score_threshold = 0.1
    soft_nms_sigma = 1.5
    name = "nms_soft_high_sigma"
    list_of_inputs.append({
        'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size,
        'iou_threshold': iou_threshold, 'score_threshold': score_threshold,
        'soft_nms_sigma': soft_nms_sigma, 'name': name
    })

    # Case 11: Score threshold filtering (only high scores survive)
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0], [4.0, 4.0, 5.0, 5.0]], dtype=np.float32)
    scores = np.array([0.9, 0.3, 0.1], dtype=np.float32)
    max_output_size = 3
    iou_threshold = 0.5
    score_threshold = 0.5
    soft_nms_sigma = 0.0
    name = "nms_score_filter"
    list_of_inputs.append({
        'boxes': boxes, 'scores': scores, 'max_output_size': max_output_size,
        'iou_threshold': iou_threshold, 'score_threshold': score_threshold,
        'soft_nms_sigma': soft_nms_sigma, 'name': name
    })

    return list_of_inputs

generated_inputs["tf.image.non_max_suppression_with_scores"] = tf_image_non_max_suppression_with_scores_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_random_hue_inputs():
    list_of_inputs = []

    # Input 1
    image = np.random.rand(2, 2, 3).astype(np.float32)
    max_delta = 0.2
    seed = 42
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Input 2
    image = np.random.randint(0, 256, size=(10, 10, 3)).astype(np.uint8)
    max_delta = 0.1
    seed = 123
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Input 3
    image = np.random.rand(2, 5, 5, 3).astype(np.float32)
    max_delta = 0.5
    seed = 1
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Input 4
    image = np.random.rand(1, 1, 3).astype(np.float32)
    max_delta = 0.0
    seed = 0
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Input 5
    image = np.random.randint(0, 256, size=(1, 3, 3, 3)).astype(np.uint8)
    max_delta = 0.3
    seed = 999
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Input 6
    image = np.random.rand(100, 100, 3).astype(np.float32)
    max_delta = 0.45
    seed = 88
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Input 7
    image = np.random.rand(5, 8, 8, 3).astype(np.float32)
    max_delta = 0.05
    seed = 7
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Input 8
    image = np.random.randint(0, 256, size=(32, 32, 3)).astype(np.uint8)
    max_delta = 0.25
    seed = 456
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Input 9
    image = np.random.randint(0, 256, size=(3, 16, 16, 3)).astype(np.uint8)
    max_delta = 0.15
    seed = 777
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Input 10
    image = np.random.rand(4, 4, 3).astype(np.float32)
    max_delta = 0.35
    seed = 11
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    return list_of_inputs

generated_inputs["tf.image.random_hue"] = tf_image_random_hue_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_random_jpeg_quality_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'image': np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8),
        'min_jpeg_quality': 50,
        'max_jpeg_quality': 90,
        'seed': 42
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'image': np.random.randint(0, 256, size=(50, 50, 1), dtype=np.uint8),
        'min_jpeg_quality': 10,
        'max_jpeg_quality': 20,
        'seed': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'image': (np.random.rand(200, 150, 3) * 255.0).astype(np.float32),
        'min_jpeg_quality': 75,
        'max_jpeg_quality': 95,
        'seed': 1234
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'image': np.random.rand(10, 10, 1).astype(np.float32),
        'min_jpeg_quality': 0,
        'max_jpeg_quality': 100,
        'seed': 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'image': np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8),
        'min_jpeg_quality': 30,
        'max_jpeg_quality': 60,
        'seed': 99
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'image': np.random.randint(0, 256, size=(32, 32, 1), dtype=np.uint8),
        'min_jpeg_quality': 80,
        'max_jpeg_quality': 90,
        'seed': 111
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'image': (np.random.rand(128, 128, 3) * 255.0).astype(np.float32),
        'min_jpeg_quality': 45,
        'max_jpeg_quality': 55,
        'seed': 7
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'image': np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8),
        'min_jpeg_quality': 20,
        'max_jpeg_quality': 80,
        'seed': 456
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'image': np.random.randint(0, 256, size=(3, 3, 1), dtype=np.uint8),
        'min_jpeg_quality': 1,
        'max_jpeg_quality': 2,
        'seed': 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'image': (np.random.rand(512, 512, 3) * 255.0).astype(np.float32),
        'min_jpeg_quality': 90,
        'max_jpeg_quality': 100,
        'seed': 888
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.image.random_jpeg_quality"] = tf_image_random_jpeg_quality_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_image_resize_inputs():
    list_of_inputs = []

    # Input 1: Standard 4-D float32 batch, bilinear, no antialias, size larger (upsampling)
    images = np.random.rand(2, 10, 10, 3).astype(np.float32)
    size = np.array([20, 20], dtype=np.int32)
    method = "bilinear"
    preserve_aspect_ratio = False
    antialias = False
    name = "resize_1"
    
    list_of_inputs.append({
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    })

    # Input 2: 3-D uint8 image, bicubic, preserve aspect ratio, size smaller
    images = np.random.randint(0, 256, size=(15, 30, 3)).astype(np.uint8)
    size = np.array([10, 10], dtype=np.int32)
    method = "bicubic"
    preserve_aspect_ratio = True
    antialias = False
    name = "resize_2"

    list_of_inputs.append({
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    })

    # Input 3: 4-D int32 batch, nearest neighbor, antialias has no effect
    images = np.random.randint(-100, 100, size=(1, 5, 5, 1)).astype(np.int32)
    size = np.array([8, 8], dtype=np.int32)
    method = "nearest"
    preserve_aspect_ratio = False
    antialias = False
    name = "resize_3"

    list_of_inputs.append({
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    })

    # Input 4: 3-D float64 image, lanczos3, antialias=True for downsampling
    images = np.random.rand(32, 32, 4).astype(np.float64)
    size = np.array([16, 16], dtype=np.int32)
    method = "lanczos3"
    preserve_aspect_ratio = False
    antialias = True
    name = "resize_4"

    list_of_inputs.append({
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    })

    # Input 5: 4-D float16, area method, preserve aspect ratio
    images = np.random.rand(4, 24, 16, 3).astype(np.float16)
    size = np.array([12, 12], dtype=np.int32)
    method = "area"
    preserve_aspect_ratio = True
    antialias = False
    name = "resize_5"

    list_of_inputs.append({
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    })

    # Input 6: 3-D float32 single channel, gaussian, preserve aspect ratio
    images = np.random.rand(100, 50, 1).astype(np.float32)
    size = np.array([40, 40], dtype=np.int32)
    method = "gaussian"
    preserve_aspect_ratio = True
    antialias = True
    name = "resize_6"

    list_of_inputs.append({
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    })

    # Input 7: 4-D float32, mitchellcubic, downsampling
    images = np.random.rand(2, 64, 64, 3).astype(np.float32)
    size = np.array([32, 32], dtype=np.int32)
    method = "mitchellcubic"
    preserve_aspect_ratio = False
    antialias = True
    name = "resize_7"

    list_of_inputs.append({
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    })

    # Input 8: 3-D uint8, lanczos5, no aspect ratio preservation
    images = np.random.randint(0, 256, size=(40, 40, 3)).astype(np.uint8)
    size = np.array([80, 80], dtype=np.int32)
    method = "lanczos5"
    preserve_aspect_ratio = False
    antialias = False
    name = "resize_8"

    list_of_inputs.append({
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    })

    # Input 9: 4-D float32, bilinear with antialias and aspect ratio preserved
    images = np.random.rand(1, 128, 64, 3).astype(np.float32)
    size = np.array([32, 32], dtype=np.int32)
    method = "bilinear"
    preserve_aspect_ratio = True
    antialias = True
    name = "resize_9"

    list_of_inputs.append({
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    })

    # Input 10: 3-D int32, nearest neighbor, upsampling
    images = np.random.randint(-10, 10, size=(4, 4, 2)).astype(np.int32)
    size = np.array([12, 12], dtype=np.int32)
    method = "nearest"
    preserve_aspect_ratio = False
    antialias = False
    name = "resize_10"

    list_of_inputs.append({
        "images": images,
        "size": size,
        "method": method,
        "preserve_aspect_ratio": preserve_aspect_ratio,
        "antialias": antialias,
        "name": name
    })

    return list_of_inputs

generated_inputs["tf.image.resize"] = tf_image_resize_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_sample_distorted_bounding_box_inputs():
    list_of_inputs = []

    # Input 1: Standard values
    input_dict_1 = {
        'image_size': np.array([224, 224, 3], dtype=np.int32),
        'bounding_boxes': np.array([[[0.1, 0.1, 0.9, 0.9]]], dtype=np.float32),
        'seed': 42,
        'min_object_covered': 0.1,
        'aspect_ratio_range': [0.75, 1.33],
        'area_range': [0.05, 1.0],
        'max_attempts': 100,
        'use_image_if_no_bounding_boxes': False,
        'name': "sample_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Larger image and multiple bounding boxes
    input_dict_2 = {
        'image_size': np.array([512, 512, 3], dtype=np.int32),
        'bounding_boxes': np.array([[[0.1, 0.1, 0.5, 0.5], [0.5, 0.5, 0.9, 0.9]]], dtype=np.float32),
        'seed': 10,
        'min_object_covered': 0.5,
        'aspect_ratio_range': [0.5, 2.0],
        'area_range': [0.1, 0.9],
        'max_attempts': 50,
        'use_image_if_no_bounding_boxes': True,
        'name': "sample_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: int64 image_size, min_object_covered=0.0, non-zero seed
    input_dict_3 = {
        'image_size': np.array([256, 256, 1], dtype=np.int64),
        'bounding_boxes': np.array([[[0.2, 0.2, 0.8, 0.8]]], dtype=np.float32),
        'seed': 3,
        'min_object_covered': 0.0,
        'aspect_ratio_range': [0.8, 1.2],
        'area_range': [0.2, 0.8],
        'max_attempts': 200,
        'use_image_if_no_bounding_boxes': False,
        'name': "sample_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: uint8 image_size, empty bounding_boxes, use_image_if_no_bounding_boxes=True
    input_dict_4 = {
        'image_size': np.array([64, 64, 3], dtype=np.uint8),
        'bounding_boxes': np.zeros((1, 0, 4), dtype=np.float32),
        'seed': 1234,
        'min_object_covered': 0.1,
        'aspect_ratio_range': [0.75, 1.33],
        'area_range': [0.05, 1.0],
        'max_attempts': 100,
        'use_image_if_no_bounding_boxes': True,
        'name': "sample_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Different aspect ratio range
    input_dict_5 = {
        'image_size': np.array([480, 640, 3], dtype=np.int32),
        'bounding_boxes': np.array([[[0.0, 0.0, 1.0, 1.0]]], dtype=np.float32),
        'seed': 7,
        'min_object_covered': 0.3,
        'aspect_ratio_range': [1.0, 1.5],
        'area_range': [0.3, 0.7],
        'max_attempts': 80,
        'use_image_if_no_bounding_boxes': False,
        'name': "sample_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Multiple bounding boxes with tight area range
    input_dict_6 = {
        'image_size': np.array([128, 128, 4], dtype=np.int32),
        'bounding_boxes': np.array([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.1, 0.1, 0.9, 0.9]]], dtype=np.float32),
        'seed': 99,
        'min_object_covered': 0.9,
        'aspect_ratio_range': [0.9, 1.1],
        'area_range': [0.8, 1.0],
        'max_attempts': 150,
        'use_image_if_no_bounding_boxes': False,
        'name': "sample_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: int16 image_size
    input_dict_7 = {
        'image_size': np.array([32, 32, 3], dtype=np.int16),
        'bounding_boxes': np.array([[[0.25, 0.25, 0.75, 0.75]]], dtype=np.float32),
        'seed': 43,
        'min_object_covered': 0.2,
        'aspect_ratio_range': [0.5, 1.5],
        'area_range': [0.1, 0.5],
        'max_attempts': 10,
        'use_image_if_no_bounding_boxes': True,
        'name': "sample_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: int8 image_size
    input_dict_8 = {
        'image_size': np.array([16, 16, 1], dtype=np.int8),
        'bounding_boxes': np.array([[[0.0, 0.0, 0.5, 0.5]]], dtype=np.float32),
        'seed': 1,
        'min_object_covered': 0.05,
        'aspect_ratio_range': [0.6, 1.4],
        'area_range': [0.4, 0.9],
        'max_attempts': 300,
        'use_image_if_no_bounding_boxes': False,
        'name': "sample_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: High max_attempts, wide range
    input_dict_9 = {
        'image_size': np.array([1000, 1000, 3], dtype=np.int32),
        'bounding_boxes': np.array([[[0.15, 0.25, 0.75, 0.85]]], dtype=np.float32),
        'seed': 999,
        'min_object_covered': 0.45,
        'aspect_ratio_range': [0.1, 10.0],
        'area_range': [0.01, 1.0],
        'max_attempts': 500,
        'use_image_if_no_bounding_boxes': True,
        'name': "sample_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Square aspect ratio constraint
    input_dict_10 = {
        'image_size': np.array([224, 224, 3], dtype=np.int32),
        'bounding_boxes': np.array([[[0.2, 0.2, 0.8, 0.8]]], dtype=np.float32),
        'seed': 123,
        'min_object_covered': 0.8,
        'aspect_ratio_range': [1.0, 1.0],
        'area_range': [0.5, 0.5],
        'max_attempts': 20,
        'use_image_if_no_bounding_boxes': False,
        'name': "sample_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.image.sample_distorted_bounding_box"] = tf_image_sample_distorted_bounding_box_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_ssim_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'img1': np.random.rand(1, 12, 12, 3).astype(np.float32),
        'img2': np.random.rand(1, 12, 12, 3).astype(np.float32),
        'max_val': 1.0,
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.01,
        'k2': 0.03,
        'return_index_map': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'img1': np.random.rand(2, 20, 20, 1).astype(np.float32) * 255.0,
        'img2': np.random.rand(2, 20, 20, 1).astype(np.float32) * 255.0,
        'max_val': 255.0,
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.01,
        'k2': 0.03,
        'return_index_map': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'img1': np.random.rand(1, 15, 15, 3).astype(np.float32),
        'img2': np.random.rand(1, 15, 15, 3).astype(np.float32),
        'max_val': 1.0,
        'filter_size': 13,
        'filter_sigma': 2.0,
        'k1': 0.01,
        'k2': 0.03,
        'return_index_map': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'img1': np.random.rand(1, 8, 8, 3).astype(np.float32),
        'img2': np.random.rand(1, 8, 8, 3).astype(np.float32),
        'max_val': 1.0,
        'filter_size': 7,
        'filter_sigma': 1.1,
        'k1': 0.02,
        'k2': 0.04,
        'return_index_map': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'img1': np.random.rand(4, 16, 16, 4).astype(np.float32),
        'img2': np.random.rand(4, 16, 16, 4).astype(np.float32),
        'max_val': 1.0,
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.01,
        'k2': 0.03,
        'return_index_map': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'img1': np.random.rand(1, 11, 11, 3).astype(np.float32),
        'img2': np.random.rand(1, 11, 11, 3).astype(np.float32),
        'max_val': 1.0,
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.05,
        'k2': 0.15,
        'return_index_map': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'img1': np.random.rand(3, 32, 32, 3).astype(np.float32) * 100.0,
        'img2': np.random.rand(3, 32, 32, 3).astype(np.float32) * 100.0,
        'max_val': 100.0,
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.01,
        'k2': 0.03,
        'return_index_map': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'img1': np.random.rand(2, 14, 14, 2).astype(np.float32),
        'img2': np.random.rand(2, 14, 14, 2).astype(np.float32),
        'max_val': 1.0,
        'filter_size': 9,
        'filter_sigma': 1.2,
        'k1': 0.01,
        'k2': 0.03,
        'return_index_map': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'img1': np.random.rand(1, 25, 25, 3).astype(np.float32),
        'img2': np.random.rand(1, 25, 25, 3).astype(np.float32),
        'max_val': 1.0,
        'filter_size': 21,
        'filter_sigma': 3.0,
        'k1': 0.01,
        'k2': 0.03,
        'return_index_map': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'img1': np.random.rand(1, 11, 11, 1).astype(np.float32) * 2.0,
        'img2': np.random.rand(1, 11, 11, 1).astype(np.float32) * 2.0,
        'max_val': 2.0,
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.005,
        'k2': 0.01,
        'return_index_map': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.image.ssim"] = tf_image_ssim_inputs()

import tensorflow as tf
import numpy as np
import copy

def gen_ssim_multiscale_inputs():
    np.random.seed(42)
    list_of_inputs = []

    # Input 1: Standard float32 images, default parameters, max_val=255.0
    img1 = np.random.uniform(0.001, 255.0, (1, 256, 256, 3)).astype(np.float32)
    img2 = np.random.uniform(0.001, 255.0, (1, 256, 256, 3)).astype(np.float32)
    input_dict = {
        'img1': img1,
        'img2': img2,
        'max_val': 255.0,
        'power_factors': (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.01,
        'k2': 0.03
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Grayscale-like float32 images, max_val=1.0, batch size of 2
    img1 = np.random.uniform(0.001, 1.0, (2, 256, 256, 1)).astype(np.float32)
    img2 = np.random.uniform(0.001, 1.0, (2, 256, 256, 1)).astype(np.float32)
    input_dict = {
        'img1': img1,
        'img2': img2,
        'max_val': 1.0,
        'power_factors': (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.01,
        'k2': 0.03
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Smaller images with 2 scales (custom power factors)
    img1 = np.random.uniform(0.001, 255.0, (1, 64, 64, 3)).astype(np.float32)
    img2 = np.random.uniform(0.001, 255.0, (1, 64, 64, 3)).astype(np.float32)
    input_dict = {
        'img1': img1,
        'img2': img2,
        'max_val': 255.0,
        'power_factors': (0.5, 0.5),
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.01,
        'k2': 0.03
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different filter size and filter sigma
    img1 = np.random.uniform(0.001, 255.0, (1, 300, 300, 3)).astype(np.float32)
    img2 = np.random.uniform(0.001, 255.0, (1, 300, 300, 3)).astype(np.float32)
    input_dict = {
        'img1': img1,
        'img2': img2,
        'max_val': 255.0,
        'power_factors': (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        'filter_size': 7,
        'filter_sigma': 1.1,
        'k1': 0.01,
        'k2': 0.03
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Custom k1 and k2 values
    img1 = np.random.uniform(0.001, 1.0, (1, 256, 256, 3)).astype(np.float32)
    img2 = np.random.uniform(0.001, 1.0, (1, 256, 256, 3)).astype(np.float32)
    input_dict = {
        'img1': img1,
        'img2': img2,
        'max_val': 1.0,
        'power_factors': (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.02,
        'k2': 0.04
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Higher batch dimensions (5D tensor shape)
    img1 = np.random.uniform(0.001, 1.0, (2, 3, 256, 256, 3)).astype(np.float32)
    img2 = np.random.uniform(0.001, 1.0, (2, 3, 256, 256, 3)).astype(np.float32)
    input_dict = {
        'img1': img1,
        'img2': img2,
        'max_val': 1.0,
        'power_factors': (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.01,
        'k2': 0.03
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcasting batch dimensions (img1 has 1 batch, img2 has 5 batches)
    img1 = np.random.uniform(0.001, 255.0, (1, 256, 256, 3)).astype(np.float32)
    img2 = np.random.uniform(0.001, 255.0, (5, 256, 256, 3)).astype(np.float32)
    input_dict = {
        'img1': img1,
        'img2': img2,
        'max_val': 255.0,
        'power_factors': (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.01,
        'k2': 0.03
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger image size with 6 scales (custom power factors)
    img1 = np.random.uniform(0.001, 1.0, (1, 512, 512, 1)).astype(np.float32)
    img2 = np.random.uniform(0.001, 1.0, (1, 512, 512, 1)).astype(np.float32)
    input_dict = {
        'img1': img1,
        'img2': img2,
        'max_val': 1.0,
        'power_factors': (0.01, 0.09, 0.2, 0.3, 0.3, 0.1),
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.01,
        'k2': 0.03
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single-scale MS-SSIM
    img1 = np.random.uniform(0.001, 1.0, (1, 64, 64, 3)).astype(np.float32)
    img2 = np.random.uniform(0.001, 1.0, (1, 64, 64, 3)).astype(np.float32)
    input_dict = {
        'img1': img1,
        'img2': img2,
        'max_val': 1.0,
        'power_factors': (1.0,),
        'filter_size': 11,
        'filter_sigma': 1.5,
        'k1': 0.01,
        'k2': 0.03
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float64 images, larger filter size, and larger image dimension
    img1 = np.random.uniform(0.001, 255.0, (1, 512, 512, 3)).astype(np.float64)
    img2 = np.random.uniform(0.001, 255.0, (1, 512, 512, 3)).astype(np.float64)
    input_dict = {
        'img1': img1,
        'img2': img2,
        'max_val': 255.0,
        'power_factors': (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        'filter_size': 15,
        'filter_sigma': 2.0,
        'k1': 0.005,
        'k2': 0.015
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.image.ssim_multiscale"] = gen_ssim_multiscale_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_brightness_inputs():
    list_of_inputs = []

    # Input 1: Basic 3D float32 image, standard seed
    image_1 = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    max_delta_1 = 0.2
    seed_1 = np.array([1, 2], dtype=np.int32)
    list_of_inputs.append({
        'image': image_1,
        'max_delta': max_delta_1,
        'seed': seed_1
    })

    # Input 2: 4D float32 batch of images, int32 seed
    image_2 = np.random.rand(2, 3, 3, 3).astype(np.float32)
    max_delta_2 = 0.5
    seed_2 = np.array([42, 24], dtype=np.int32)
    list_of_inputs.append({
        'image': image_2,
        'max_delta': max_delta_2,
        'seed': seed_2
    })

    # Input 3: 3D uint8 image, int64 seed (Note: on XLA, only int32 is allowed, so using int32 is safer, but standard supports int64 too. Let's use int32 for general safety, or mix them.)
    image_3 = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)
    max_delta_3 = 10.0
    seed_3 = np.array([123, 456], dtype=np.int32)
    list_of_inputs.append({
        'image': image_3,
        'max_delta': max_delta_3,
        'seed': seed_3
    })

    # Input 4: 3D float16 grayscale image, small max_delta
    image_4 = np.random.rand(4, 4, 1).astype(np.float16)
    max_delta_4 = 0.1
    seed_4 = np.array([9, 99], dtype=np.int32)
    list_of_inputs.append({
        'image': image_4,
        'max_delta': max_delta_4,
        'seed': seed_4
    })

    # Input 5: 4D uint8 batch of grayscale images, large max_delta
    image_5 = np.random.randint(0, 256, size=(3, 5, 5, 1), dtype=np.uint8)
    max_delta_5 = 50.0
    seed_5 = np.array([7, 7], dtype=np.int32)
    list_of_inputs.append({
        'image': image_5,
        'max_delta': max_delta_5,
        'seed': seed_5
    })

    # Input 6: Zero max_delta (no brightness adjustment), 3D float32 image
    image_6 = np.random.rand(100, 100, 3).astype(np.float32)
    max_delta_6 = 0.0
    seed_6 = np.array([0, 0], dtype=np.int32)
    list_of_inputs.append({
        'image': image_6,
        'max_delta': max_delta_6,
        'seed': seed_6
    })

    # Input 7: 4D float32 RGBA images (4 channels)
    image_7 = np.random.rand(2, 10, 10, 4).astype(np.float32)
    max_delta_7 = 0.3
    seed_7 = np.array([11, 22], dtype=np.int32)
    list_of_inputs.append({
        'image': image_7,
        'max_delta': max_delta_7,
        'seed': seed_7
    })

    # Input 8: Minimal sized 3D float32 image (1x1x1)
    image_8 = np.array([[[0.5]]], dtype=np.float32)
    max_delta_8 = 1.5
    seed_8 = np.array([1000, 2000], dtype=np.int32)
    list_of_inputs.append({
        'image': image_8,
        'max_delta': max_delta_8,
        'seed': seed_8
    })

    # Input 9: 4D uint8 batch, standard seed
    image_9 = np.random.randint(0, 256, size=(4, 8, 8, 3), dtype=np.uint8)
    max_delta_9 = 25.0
    seed_9 = np.array([111, 222], dtype=np.int32)
    list_of_inputs.append({
        'image': image_9,
        'max_delta': max_delta_9,
        'seed': seed_9
    })

    # Input 10: 3D float32 image, small max_delta
    image_10 = np.random.rand(5, 5, 3).astype(np.float32)
    max_delta_10 = 0.05
    seed_10 = np.array([12345, 67890], dtype=np.int32)
    list_of_inputs.append({
        'image': image_10,
        'max_delta': max_delta_10,
        'seed': seed_10
    })

    return list_of_inputs

generated_inputs["tf.image.stateless_random_brightness"] = tf_image_stateless_random_brightness_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_contrast_inputs():
    list_of_inputs = []

    # Input 1: Basic Float32, 3D
    list_of_inputs.append({
        'image': np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                           [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32),
        'lower': 0.2,
        'upper': 0.5,
        'seed': np.array([1, 2], dtype=np.int32)
    })

    # Input 2: Large dimensions, Float32, 3D
    list_of_inputs.append({
        'image': np.random.rand(10, 10, 3).astype(np.float32),
        'lower': 0.1,
        'upper': 0.9,
        'seed': np.array([42, 43], dtype=np.int64)
    })

    # Input 3: Float32, 1-channel, 3D
    list_of_inputs.append({
        'image': np.random.rand(5, 5, 1).astype(np.float32),
        'lower': 0.0,
        'upper': 1.0,
        'seed': np.array([0, 0], dtype=np.int32)
    })

    # Input 4: Float32, 3D with 3 channels
    list_of_inputs.append({
        'image': np.random.rand(4, 4, 3).astype(np.float32),
        'lower': 0.5,
        'upper': 1.5,
        'seed': np.array([123, 456], dtype=np.int32)
    })

    # Input 5: Batched Float32, 4D
    list_of_inputs.append({
        'image': np.random.rand(2, 8, 8, 3).astype(np.float32),
        'lower': 0.2,
        'upper': 1.8,
        'seed': np.array([7, 8], dtype=np.int64)
    })

    # Input 6: Float32, 3D, small size
    list_of_inputs.append({
        'image': np.random.rand(3, 3, 3).astype(np.float32),
        'lower': 0.5,
        'upper': 0.6,
        'seed': np.array([10, 20], dtype=np.int32)
    })

    # Input 7: High scale factors, Float32, 3D
    list_of_inputs.append({
        'image': np.random.rand(6, 6, 3).astype(np.float32),
        'lower': 2.0,
        'upper': 5.0,
        'seed': np.array([99, 999], dtype=np.int32)
    })

    # Input 8: 5D Tensor, Float32
    list_of_inputs.append({
        'image': np.random.rand(2, 2, 4, 4, 3).astype(np.float32),
        'lower': 0.8,
        'upper': 1.2,
        'seed': np.array([1, 1], dtype=np.int64)
    })

    # Input 9: Batched Float32, 4D, single channel
    list_of_inputs.append({
        'image': np.random.rand(1, 16, 16, 1).astype(np.float32),
        'lower': 0.0,
        'upper': 2.0,
        'seed': np.array([4, 2], dtype=np.int32)
    })

    # Input 10: Float32 image, 3D, very close bounds
    list_of_inputs.append({
        'image': np.random.rand(3, 3, 3).astype(np.float32),
        'lower': 1.0,
        'upper': 1.0001,
        'seed': np.array([11, 22], dtype=np.int32)
    })

    return list_of_inputs

generated_inputs["tf.image.stateless_random_contrast"] = tf_image_stateless_random_contrast_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_flip_left_right_inputs():
    list_of_inputs = []
    
    # Input 1: 3-D float32 image, standard seed
    list_of_inputs.append({
        'image': np.random.rand(2, 2, 1).astype(np.float32),
        'seed': np.array([1, 2], dtype=np.int32)
    })
    
    # Input 2: 3-D uint8 image, different seed
    list_of_inputs.append({
        'image': np.random.randint(0, 256, (3, 4, 3), dtype=np.uint8),
        'seed': np.array([42, 24], dtype=np.int32)
    })
    
    # Input 3: 4-D float32 image, standard seed
    list_of_inputs.append({
        'image': np.random.rand(1, 2, 2, 1).astype(np.float32),
        'seed': np.array([10, 20], dtype=np.int32)
    })
    
    # Input 4: 3-D int32 image with negative values
    list_of_inputs.append({
        'image': np.random.randint(-100, 100, (3, 3, 3), dtype=np.int32),
        'seed': np.array([0, 0], dtype=np.int32)
    })
    
    # Input 5: 4-D int32 image, negative values
    list_of_inputs.append({
        'image': np.random.randint(-50, 50, (2, 5, 5, 3), dtype=np.int32),
        'seed': np.array([5, 10], dtype=np.int32)
    })
    
    # Input 6: 3-D float32 image with 1 channel
    list_of_inputs.append({
        'image': np.random.rand(4, 4, 1).astype(np.float32),
        'seed': np.array([99, 99], dtype=np.int32)
    })
    
    # Input 7: 3-D float32 minimal image size
    list_of_inputs.append({
        'image': np.random.rand(1, 1, 1).astype(np.float32),
        'seed': np.array([123, 456], dtype=np.int32)
    })
    
    # Input 8: 4-D int32 image with single pixel height
    list_of_inputs.append({
        'image': np.random.randint(0, 10, (2, 1, 3, 2), dtype=np.int32),
        'seed': np.array([7, 8], dtype=np.int32)
    })
    
    # Input 9: 3-D float32 negative values only
    list_of_inputs.append({
        'image': (np.random.rand(3, 2, 2) * -10).astype(np.float32),
        'seed': np.array([100, 200], dtype=np.int32)
    })
    
    # Input 10: 4-D uint8 image
    list_of_inputs.append({
        'image': np.random.randint(0, 256, (2, 2, 2, 2), dtype=np.uint8),
        'seed': np.array([1000, 2000], dtype=np.int32)
    })
    
    return list_of_inputs

generated_inputs["tf.image.stateless_random_flip_left_right"] = tf_image_stateless_random_flip_left_right_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_flip_up_down_inputs():
    list_of_inputs = []

    # Input 1: 3-D float32 image, int32 seed
    image_1 = np.random.rand(4, 4, 3).astype(np.float32)
    seed_1 = np.array([1, 2], dtype=np.int32)
    list_of_inputs.append({"image": image_1, "seed": seed_1})

    # Input 2: 3-D float32 image, int64 seed
    image_2 = np.random.rand(8, 8, 3).astype(np.float32)
    seed_2 = np.array([42, 43], dtype=np.int64)
    list_of_inputs.append({"image": image_2, "seed": seed_2})

    # Input 3: 4-D float32 image (batch size 2), int32 seed
    image_3 = np.random.rand(2, 6, 6, 1).astype(np.float32)
    seed_3 = np.array([123, 456], dtype=np.int32)
    list_of_inputs.append({"image": image_3, "seed": seed_3})

    # Input 4: 4-D int32 image, int32 seed
    image_4 = np.random.randint(-100, 100, size=(1, 5, 5, 3)).astype(np.int32)
    seed_4 = np.array([7, 8], dtype=np.int32)
    list_of_inputs.append({"image": image_4, "seed": seed_4})

    # Input 5: 3-D float64 image, int64 seed
    image_5 = np.random.rand(10, 10, 4).astype(np.float64)
    seed_5 = np.array([99, 100], dtype=np.int64)
    list_of_inputs.append({"image": image_5, "seed": seed_5})

    # Input 6: 4-D int32 image, int32 seed
    image_6 = np.random.randint(-1000, 1000, size=(3, 3, 3, 2)).astype(np.int32)
    seed_6 = np.array([0, 0], dtype=np.int32)
    list_of_inputs.append({"image": image_6, "seed": seed_6})

    # Input 7: 3-D float32 image, int64 seed
    image_7 = np.random.rand(1, 1, 1).astype(np.float32)
    seed_7 = np.array([-1, -2], dtype=np.int64)
    list_of_inputs.append({"image": image_7, "seed": seed_7})

    # Input 8: 4-D float32 image, int32 seed
    image_8 = np.random.rand(4, 16, 16, 3).astype(np.float32)
    seed_8 = np.array([9999, 8888], dtype=np.int32)
    list_of_inputs.append({"image": image_8, "seed": seed_8})

    # Input 9: 3-D float64 image, int32 seed
    image_9 = np.random.rand(7, 7, 2).astype(np.float64)
    seed_9 = np.array([111, 222], dtype=np.int32)
    list_of_inputs.append({"image": image_9, "seed": seed_9})

    # Input 10: 4-D int64 image, int64 seed
    image_10 = np.random.randint(0, 100000, size=(2, 8, 8, 4)).astype(np.int64)
    seed_10 = np.array([123456789, 987654321], dtype=np.int64)
    list_of_inputs.append({"image": image_10, "seed": seed_10})

    return list_of_inputs

generated_inputs["tf.image.stateless_random_flip_up_down"] = tf_image_stateless_random_flip_up_down_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_hue_inputs():
    list_of_inputs = []

    # Case 1: 3D float32 image, max_delta 0.2, int32 seed
    image = np.random.rand(2, 2, 3).astype(np.float32)
    max_delta = 0.2
    seed = np.array([1, 2], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Case 2: 3D float32 image, max_delta 0.0 (no change), int64 seed
    image = np.random.rand(4, 4, 3).astype(np.float32)
    max_delta = 0.0
    seed = np.array([42, 42], dtype=np.int64)
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Case 3: 3D float32 image (1x1 pixel), max_delta 0.5, int32 seed
    image = np.random.rand(1, 1, 3).astype(np.float32)
    max_delta = 0.5
    seed = np.array([0, 0], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Case 4: 4D float32 image batch, max_delta 0.1, int32 seed
    image = np.random.rand(2, 3, 4, 3).astype(np.float32)
    max_delta = 0.1
    seed = np.array([99, 100], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Case 5: 3D float32 image, max_delta 0.3, int64 seed
    image = np.random.rand(1, 3, 3).astype(np.float32)
    max_delta = 0.3
    seed = np.array([123, 456], dtype=np.int64)
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Case 6: Larger 3D float32 image, max_delta 0.45, int32 seed
    image = np.random.rand(32, 32, 3).astype(np.float32)
    max_delta = 0.45
    seed = np.array([7, 8], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Case 7: 4D float32 image, max_delta 0.05, int64 seed
    image = np.random.rand(2, 2, 2, 3).astype(np.float32)
    max_delta = 0.05
    seed = np.array([11, 22], dtype=np.int64)
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Case 8: 3D float32 image, max_delta 0.15, int32 seed
    image = np.random.rand(5, 5, 3).astype(np.float32)
    max_delta = 0.15
    seed = np.array([2023, 2024], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Case 9: 3D float32 image, max_delta 0.5, negative int32 seed
    image = np.random.rand(2, 3, 3).astype(np.float32)
    max_delta = 0.5
    seed = np.array([-1, -2], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    # Case 10: 3D float32 image, max_delta 0.25, large int64 seed
    image = np.random.rand(1, 2, 3).astype(np.float32)
    max_delta = 0.25
    seed = np.array([123456, 789012], dtype=np.int64)
    list_of_inputs.append({
        'image': image,
        'max_delta': max_delta,
        'seed': seed
    })

    return list_of_inputs

generated_inputs["tf.image.stateless_random_hue"] = tf_image_stateless_random_hue_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_jpeg_quality_inputs():
    list_of_inputs = []

    # Input 1
    image = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)
    min_jpeg_quality = 75
    max_jpeg_quality = 95
    seed = np.array([1, 2], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'min_jpeg_quality': min_jpeg_quality,
        'max_jpeg_quality': max_jpeg_quality,
        'seed': seed
    })

    # Input 2
    image = np.random.randint(0, 256, size=(100, 100, 1), dtype=np.uint8)
    min_jpeg_quality = 0
    max_jpeg_quality = 10
    seed = np.array([42, 43], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'min_jpeg_quality': min_jpeg_quality,
        'max_jpeg_quality': max_jpeg_quality,
        'seed': seed
    })

    # Input 3
    image = np.random.rand(32, 32, 3).astype(np.float32)
    min_jpeg_quality = 50
    max_jpeg_quality = 60
    seed = np.array([10, 20], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'min_jpeg_quality': min_jpeg_quality,
        'max_jpeg_quality': max_jpeg_quality,
        'seed': seed
    })

    # Input 4
    image = np.random.randint(0, 256, size=(1, 1, 3), dtype=np.uint8)
    min_jpeg_quality = 90
    max_jpeg_quality = 100
    seed = np.array([100, 200], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'min_jpeg_quality': min_jpeg_quality,
        'max_jpeg_quality': max_jpeg_quality,
        'seed': seed
    })

    # Input 5
    image = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    min_jpeg_quality = 30
    max_jpeg_quality = 80
    seed = np.array([-5, 5], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'min_jpeg_quality': min_jpeg_quality,
        'max_jpeg_quality': max_jpeg_quality,
        'seed': seed
    })

    # Input 6
    image = np.random.rand(64, 64, 1).astype(np.float32)
    min_jpeg_quality = 10
    max_jpeg_quality = 90
    seed = np.array([1000, 2000], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'min_jpeg_quality': min_jpeg_quality,
        'max_jpeg_quality': max_jpeg_quality,
        'seed': seed
    })

    # Input 7
    image = np.random.randint(0, 256, size=(5, 5, 3), dtype=np.uint8)
    min_jpeg_quality = 0
    max_jpeg_quality = 100
    seed = np.array([0, 0], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'min_jpeg_quality': min_jpeg_quality,
        'max_jpeg_quality': max_jpeg_quality,
        'seed': seed
    })

    # Input 8
    image = np.random.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)
    min_jpeg_quality = 45
    max_jpeg_quality = 55
    seed = np.array([11, 22], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'min_jpeg_quality': min_jpeg_quality,
        'max_jpeg_quality': max_jpeg_quality,
        'seed': seed
    })

    # Input 9
    image = np.random.randint(0, 256, size=(50, 50, 1), dtype=np.uint8)
    min_jpeg_quality = 20
    max_jpeg_quality = 40
    seed = np.array([99, 99], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'min_jpeg_quality': min_jpeg_quality,
        'max_jpeg_quality': max_jpeg_quality,
        'seed': seed
    })

    # Input 10
    image = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
    min_jpeg_quality = 80
    max_jpeg_quality = 85
    seed = np.array([1234, 5678], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'min_jpeg_quality': min_jpeg_quality,
        'max_jpeg_quality': max_jpeg_quality,
        'seed': seed
    })

    return list_of_inputs

generated_inputs["tf.image.stateless_random_jpeg_quality"] = tf_image_stateless_random_jpeg_quality_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_saturation_inputs():
    list_of_inputs = []
    
    # Input 1
    image = np.random.rand(2, 2, 3).astype(np.float32)
    lower = 0.5
    upper = 1.0
    seed = np.array([1, 2], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'lower': lower,
        'upper': upper,
        'seed': seed
    })

    # Input 2
    image = np.random.rand(3, 3, 3).astype(np.float32)
    lower = 0.1
    upper = 0.9
    seed = np.array([42, 43], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'lower': lower,
        'upper': upper,
        'seed': seed
    })

    # Input 3
    image = np.random.randint(0, 256, size=(4, 4, 3)).astype(np.uint8)
    lower = 0.0
    upper = 2.0
    seed = np.array([10, 20], dtype=np.int64)
    list_of_inputs.append({
        'image': image,
        'lower': lower,
        'upper': upper,
        'seed': seed
    })

    # Input 4
    image = np.random.rand(2, 2, 2, 3).astype(np.float64)
    lower = 0.2
    upper = 0.8
    seed = np.array([9, 9], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'lower': lower,
        'upper': upper,
        'seed': seed
    })

    # Input 5
    image = np.random.rand(5, 5, 3).astype(np.float32)
    lower = 1.0
    upper = 3.0
    seed = np.array([0, 1], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'lower': lower,
        'upper': upper,
        'seed': seed
    })

    # Input 6
    image = np.random.rand(1, 1, 3).astype(np.float32)
    lower = 0.0
    upper = 1.0
    seed = np.array([123, 456], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'lower': lower,
        'upper': upper,
        'seed': seed
    })

    # Input 7
    image = np.random.rand(2, 3, 4, 3).astype(np.float32)
    lower = 0.5
    upper = 0.6
    seed = np.array([100, 200], dtype=np.int64)
    list_of_inputs.append({
        'image': image,
        'lower': lower,
        'upper': upper,
        'seed': seed
    })

    # Input 8
    image = np.random.randint(0, 256, size=(5, 5, 3)).astype(np.uint8)
    lower = 1.2
    upper = 1.8
    seed = np.array([7, 8], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'lower': lower,
        'upper': upper,
        'seed': seed
    })

    # Input 9
    image = np.random.rand(2, 1, 1, 3).astype(np.float32)
    lower = 0.1
    upper = 10.0
    seed = np.array([1234, 5678], dtype=np.int32)
    list_of_inputs.append({
        'image': image,
        'lower': lower,
        'upper': upper,
        'seed': seed
    })

    # Input 10
    image = np.random.rand(3, 3, 3).astype(np.float64)
    lower = 0.0
    upper = 0.5
    seed = np.array([999, 999], dtype=np.int64)
    list_of_inputs.append({
        'image': image,
        'lower': lower,
        'upper': upper,
        'seed': seed
    })

    return list_of_inputs

generated_inputs["tf.image.stateless_random_saturation"] = tf_image_stateless_random_saturation_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_tensor_diag_part_inputs():
    list_of_inputs = []

    list_of_inputs.append({
        'input': np.random.randn(3, 3).astype(np.float32),
        'name': 'diag_1'
    })

    list_of_inputs.append({
        'input': np.random.randint(-10, 10, size=(1, 1)).astype(np.int32),
        'name': 'diag_2'
    })

    list_of_inputs.append({
        'input': np.random.uniform(-5.0, 5.0, size=(5, 5)).astype(np.float64),
        'name': 'diag_3'
    })

    list_of_inputs.append({
        'input': np.random.randint(-5, 5, size=(2, 2, 2, 2)).astype(np.int64),
        'name': 'diag_4'
    })

    list_of_inputs.append({
        'input': np.random.randn(3, 4, 3, 4).astype(np.float32),
        'name': 'diag_5'
    })

    list_of_inputs.append({
        'input': np.random.randn(1, 5, 1, 5).astype(np.float32),
        'name': 'diag_6'
    })

    list_of_inputs.append({
        'input': np.random.randint(0, 100, size=(2, 2, 2, 2, 2, 2)).astype(np.int32),
        'name': 'diag_7'
    })

    list_of_inputs.append({
        'input': np.random.randn(2, 3, 1, 2, 3, 1).astype(np.float64),
        'name': 'diag_8'
    })

    real = np.random.randn(10, 10).astype(np.float32)
    imag = np.random.randn(10, 10).astype(np.float32)
    list_of_inputs.append({
        'input': (real + 1j * imag),
        'name': 'diag_9'
    })

    list_of_inputs.append({
        'input': np.random.randn(2, 3, 2, 3).astype(np.float16),
        'name': 'diag_10'
    })

    return list_of_inputs

generated_inputs["tf.linalg.tensor_diag_part"] = tf_linalg_tensor_diag_part_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_tridiagonal_matmul_inputs():
    list_of_inputs = []

    # Input 1: Float32, no batch, M=3, N=2
    superdiag = np.array([1.0, 2.0, 0.0], dtype=np.float32)
    maindiag = np.array([3.0, 4.0, 5.0], dtype=np.float32)
    subdiag = np.array([0.0, 6.0, 7.0], dtype=np.float32)
    rhs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    input_dict = {
        'diagonals': (superdiag, maindiag, subdiag),
        'rhs': rhs,
        'diagonals_format': 'sequence',
        'name': 'matmul_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float64, batch [2], M=4, N=1
    superdiag = np.random.randn(2, 4).astype(np.float64)
    maindiag = np.random.randn(2, 4).astype(np.float64)
    subdiag = np.random.randn(2, 4).astype(np.float64)
    rhs = np.random.randn(2, 4, 1).astype(np.float64)
    input_dict = {
        'diagonals': (superdiag, maindiag, subdiag),
        'rhs': rhs,
        'diagonals_format': 'sequence',
        'name': 'matmul_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex64, batch [3, 2], M=2, N=3
    superdiag = (np.random.randn(3, 2, 2) + 1j * np.random.randn(3, 2, 2)).astype(np.complex64)
    maindiag = (np.random.randn(3, 2, 2) + 1j * np.random.randn(3, 2, 2)).astype(np.complex64)
    subdiag = (np.random.randn(3, 2, 2) + 1j * np.random.randn(3, 2, 2)).astype(np.complex64)
    rhs = (np.random.randn(3, 2, 2, 3) + 1j * np.random.randn(3, 2, 2, 3)).astype(np.complex64)
    input_dict = {
        'diagonals': (superdiag, maindiag, subdiag),
        'rhs': rhs,
        'diagonals_format': 'sequence',
        'name': 'matmul_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex128, no batch, M=5, N=5
    superdiag = np.zeros(5, dtype=np.complex128)
    maindiag = np.ones(5, dtype=np.complex128)
    subdiag = np.zeros(5, dtype=np.complex128)
    rhs = np.eye(5, dtype=np.complex128)
    input_dict = {
        'diagonals': (superdiag, maindiag, subdiag),
        'rhs': rhs,
        'diagonals_format': 'sequence',
        'name': 'matmul_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float32, batch [1], M=1, N=1
    superdiag = np.array([[0.0]], dtype=np.float32)
    maindiag = np.array([[5.0]], dtype=np.float32)
    subdiag = np.array([[0.0]], dtype=np.float32)
    rhs = np.array([[[2.0]]], dtype=np.float32)
    input_dict = {
        'diagonals': (superdiag, maindiag, subdiag),
        'rhs': rhs,
        'diagonals_format': 'sequence',
        'name': 'matmul_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float64, negative values, M=3, N=3
    superdiag = np.array([-1.5, -2.5, 0.0], dtype=np.float64)
    maindiag = np.array([-3.5, -4.5, -5.5], dtype=np.float64)
    subdiag = np.array([0.0, -6.5, -7.5], dtype=np.float64)
    rhs = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 2.0], [-3.0, 3.0, -3.0]], dtype=np.float64)
    input_dict = {
        'diagonals': (superdiag, maindiag, subdiag),
        'rhs': rhs,
        'diagonals_format': 'sequence',
        'name': 'matmul_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float32, batch [2], M=10, N=5
    superdiag = np.random.uniform(-1, 1, (2, 10)).astype(np.float32)
    maindiag = np.random.uniform(-1, 1, (2, 10)).astype(np.float32)
    subdiag = np.random.uniform(-1, 1, (2, 10)).astype(np.float32)
    rhs = np.random.uniform(-1, 1, (2, 10, 5)).astype(np.float32)
    input_dict = {
        'diagonals': (superdiag, maindiag, subdiag),
        'rhs': rhs,
        'diagonals_format': 'sequence',
        'name': 'matmul_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex64, zeros, M=4, N=2
    superdiag = np.zeros((4,), dtype=np.complex64)
    maindiag = np.zeros((4,), dtype=np.complex64)
    subdiag = np.zeros((4,), dtype=np.complex64)
    rhs = np.zeros((4, 2), dtype=np.complex64)
    input_dict = {
        'diagonals': (superdiag, maindiag, subdiag),
        'rhs': rhs,
        'diagonals_format': 'sequence',
        'name': 'matmul_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Float64, high-dimensional batch [2, 2, 2], M=2, N=2
    superdiag = np.random.randn(2, 2, 2, 2).astype(np.float64)
    maindiag = np.random.randn(2, 2, 2, 2).astype(np.float64)
    subdiag = np.random.randn(2, 2, 2, 2).astype(np.float64)
    rhs = np.random.randn(2, 2, 2, 2, 2).astype(np.float64)
    input_dict = {
        'diagonals': (superdiag, maindiag, subdiag),
        'rhs': rhs,
        'diagonals_format': 'sequence',
        'name': 'matmul_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float32, large identity-like values, M=3, N=3
    superdiag = np.zeros(3, dtype=np.float32)
    maindiag = np.ones(3, dtype=np.float32) * 10.0
    subdiag = np.zeros(3, dtype=np.float32)
    rhs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    input_dict = {
        'diagonals': (superdiag, maindiag, subdiag),
        'rhs': rhs,
        'diagonals_format': 'sequence',
        'name': 'matmul_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.tridiagonal_matmul_2"] = tf_linalg_tridiagonal_matmul_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_tridiagonal_solve_inputs():
    list_of_inputs = []
    
    # 1. Compact format, float32, no batch, M=4
    diagonals = np.array([
        [1.0, 1.0, 1.0, 0.0],
        [4.0, 4.0, 4.0, 4.0],
        [0.0, 1.0, 1.0, 1.0]
    ], dtype=np.float32)
    rhs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'transpose_rhs': False,
        'conjugate_rhs': False,
        'name': 'solve_1',
        'partial_pivoting': True,
        'perturb_singular': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2. Compact format, float64, batch [2], M=3, multiple RHS (K=2)
    diagonals = np.zeros((2, 3, 3), dtype=np.float64)
    diagonals[:, 0, :] = 1.0
    diagonals[:, 1, :] = 4.0
    diagonals[:, 2, :] = 1.0
    rhs = np.ones((2, 3, 2), dtype=np.float64)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'transpose_rhs': False,
        'conjugate_rhs': False,
        'name': 'solve_2',
        'partial_pivoting': True,
        'perturb_singular': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3. Matrix format, float32, no batch, M=3
    diagonals = np.array([
        [4.0, 1.0, 0.0],
        [1.0, 4.0, 1.0],
        [0.0, 1.0, 4.0]
    ], dtype=np.float32)
    rhs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'matrix',
        'transpose_rhs': False,
        'conjugate_rhs': False,
        'name': 'solve_3',
        'partial_pivoting': False,
        'perturb_singular': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4. Compact format, complex64, M=3, transpose_rhs=True, K=2 (rhs shape [K, M])
    diagonals = np.zeros((3, 3), dtype=np.complex64)
    diagonals[0, :] = 1.0 + 0j
    diagonals[1, :] = 4.0 + 0j
    diagonals[2, :] = 1.0 + 0j
    rhs = np.array([[1.0 + 1j, 2.0 + 2j, 3.0 + 3j], [4.0, 5.0, 6.0]], dtype=np.complex64)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'transpose_rhs': True,
        'conjugate_rhs': True,
        'name': 'solve_4',
        'partial_pivoting': True,
        'perturb_singular': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5. Matrix format, batch shape [1, 2], float32, M=3
    diagonals = np.zeros((1, 2, 3, 3), dtype=np.float32)
    for i in range(3):
        diagonals[..., i, i] = 4.0
        if i > 0:
            diagonals[..., i, i-1] = 1.0
        if i < 2:
            diagonals[..., i, i+1] = 1.0
    rhs = np.ones((1, 2, 3), dtype=np.float32)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'matrix',
        'transpose_rhs': False,
        'conjugate_rhs': False,
        'name': 'solve_5',
        'partial_pivoting': True,
        'perturb_singular': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6. Compact format, float32, batch shape [3], M=5, partial_pivoting=False
    diagonals = np.zeros((3, 3, 5), dtype=np.float32)
    diagonals[:, 0, :] = 1.0
    diagonals[:, 1, :] = 5.0
    diagonals[:, 2, :] = 1.0
    rhs = np.ones((3, 5, 2), dtype=np.float32)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'transpose_rhs': False,
        'conjugate_rhs': False,
        'name': 'solve_6',
        'partial_pivoting': False,
        'perturb_singular': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7. Compact format, complex128, M=4
    diagonals = np.zeros((3, 4), dtype=np.complex128)
    diagonals[0, :] = 1.0 + 0j
    diagonals[1, :] = 4.0 + 0j
    diagonals[2, :] = 1.0 + 0j
    rhs = np.array([1.0 + 1j, 2.0 + 2j, 3.0 + 3j, 4.0 + 4j], dtype=np.complex128)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'transpose_rhs': False,
        'conjugate_rhs': True,
        'name': 'solve_7',
        'partial_pivoting': True,
        'perturb_singular': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8. Matrix format, float64, batch [2], M=3, RHS [2, 3]
    diagonals = np.zeros((2, 3, 3), dtype=np.float64)
    for b in range(2):
        for i in range(3):
            diagonals[b, i, i] = 4.0
            if i > 0:
                diagonals[b, i, i-1] = 1.0
            if i < 2:
                diagonals[b, i, i+1] = 1.0
    rhs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'matrix',
        'transpose_rhs': False,
        'conjugate_rhs': False,
        'name': 'solve_8',
        'partial_pivoting': True,
        'perturb_singular': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9. Compact format, float32, M=2
    diagonals = np.array([
        [1.0, 0.0],
        [4.0, 4.0],
        [0.0, 1.0]
    ], dtype=np.float32)
    rhs = np.array([2.0, 3.0], dtype=np.float32)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'transpose_rhs': False,
        'conjugate_rhs': False,
        'name': 'solve_9',
        'partial_pivoting': False,
        'perturb_singular': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10. Matrix format, float32, M=4, transpose_rhs=True, K=2 (rhs shape [2, 4])
    diagonals = np.zeros((4, 4), dtype=np.float32)
    for i in range(4):
        diagonals[i, i] = 4.0
        if i > 0:
            diagonals[i, i-1] = 1.0
        if i < 3:
            diagonals[i, i+1] = 1.0
    rhs = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'matrix',
        'transpose_rhs': True,
        'conjugate_rhs': False,
        'name': 'solve_10',
        'partial_pivoting': True,
        'perturb_singular': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.tridiagonal_solve"] = tf_linalg_tridiagonal_solve_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_in_top_k_inputs():
    list_of_inputs = []

    # Input 1: Basic example
    targets = np.array([0, 1, 3], dtype=np.int32)
    predictions = np.array([
        [1.2, -0.3, 2.8, 5.2],
        [0.1, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.3, 0.3]
    ], dtype=np.float32)
    k = 2
    name = "case1"
    list_of_inputs.append({
        'targets': targets,
        'predictions': predictions,
        'k': k,
        'name': name
    })

    # Input 2: Int64 targets and smaller batch
    targets = np.array([1, 0], dtype=np.int64)
    predictions = np.array([
        [-1.0, 2.0],
        [3.0, -1.0]
    ], dtype=np.float32)
    k = 1
    name = "case2"
    list_of_inputs.append({
        'targets': targets,
        'predictions': predictions,
        'k': k,
        'name': name
    })

    # Input 3: Larger batch with varying values
    targets = np.array([2, 2, 1, 0], dtype=np.int32)
    predictions = np.array([
        [0.1, 0.2, 0.7],
        [0.1, 0.8, 0.1],
        [0.5, 0.4, 0.1],
        [0.9, 0.0, 0.1]
    ], dtype=np.float32)
    k = 2
    name = "case3"
    list_of_inputs.append({
        'targets': targets,
        'predictions': predictions,
        'k': k,
        'name': name
    })

    # Input 4: Minimum batch size (1)
    targets = np.array([0], dtype=np.int32)
    predictions = np.array([[1.5]], dtype=np.float32)
    k = 1
    name = "case4"
    list_of_inputs.append({
        'targets': targets,
        'predictions': predictions,
        'k': k,
        'name': name
    })

    # Input 5: k equals number of classes
    targets = np.array([1, 2, 0, 1, 2], dtype=np.int32)
    predictions = np.arange(15, dtype=np.float32).reshape(5, 3)
    k = 3
    name = "case5"
    list_of_inputs.append({
        'targets': targets,
        'predictions': predictions,
        'k': k,
        'name': name
    })

    # Input 6: Negative predictions
    targets = np.array([2, 0], dtype=np.int64)
    predictions = np.array([
        [-10.0, -5.0, -1.0],
        [-1.0, -2.0, -3.0]
    ], dtype=np.float32)
    k = 1
    name = "case6"
    list_of_inputs.append({
        'targets': targets,
        'predictions': predictions,
        'k': k,
        'name': name
    })

    # Input 7: Identity targets with k = 3
    targets = np.array([0, 1, 2], dtype=np.int32)
    predictions = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=np.float32)
    k = 3
    name = "case7"
    list_of_inputs.append({
        'targets': targets,
        'predictions': predictions,
        'k': k,
        'name': name
    })

    # Input 8: One-hot predictions
    targets = np.array([4, 3, 2, 1, 0], dtype=np.int32)
    predictions = np.eye(5, dtype=np.float32)
    k = 4
    name = "case8"
    list_of_inputs.append({
        'targets': targets,
        'predictions': predictions,
        'k': k,
        'name': name
    })

    # Input 9: All identical values (ties test)
    targets = np.array([0, 1], dtype=np.int32)
    predictions = np.array([
        [5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0]
    ], dtype=np.float32)
    k = 1
    name = "case9"
    list_of_inputs.append({
        'targets': targets,
        'predictions': predictions,
        'k': k,
        'name': name
    })

    # Input 10: Large values
    targets = np.array([1, 1], dtype=np.int64)
    predictions = np.array([
        [1e5, 1e6],
        [1e6, 1e5]
    ], dtype=np.float32)
    k = 1
    name = "case10"
    list_of_inputs.append({
        'targets': targets,
        'predictions': predictions,
        'k': k,
        'name': name
    })

    return list_of_inputs

generated_inputs["tf.math.in_top_k"] = tf_math_in_top_k_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_invert_permutation_inputs():
    list_of_inputs = []
    
    # 1. 1-element permutation (int32)
    list_of_inputs.append({
        'x': np.array([0], dtype=np.int32),
        'name': 'perm_1'
    })
    
    # 2. 2-element permutation (int32)
    list_of_inputs.append({
        'x': np.array([1, 0], dtype=np.int32),
        'name': 'perm_2'
    })
    
    # 3. 3-element permutation (int32)
    list_of_inputs.append({
        'x': np.array([2, 0, 1], dtype=np.int32),
        'name': 'perm_3'
    })
    
    # 4. 5-element permutation (int32)
    list_of_inputs.append({
        'x': np.array([3, 4, 0, 2, 1], dtype=np.int32),
        'name': 'perm_4'
    })
    
    # 5. Identity permutation (int32)
    list_of_inputs.append({
        'x': np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
        'name': 'perm_5'
    })
    
    # 6. Reversed permutation (int64)
    list_of_inputs.append({
        'x': np.array([5, 4, 3, 2, 1, 0], dtype=np.int64),
        'name': 'perm_6'
    })
    
    # 7. Shift permutation (int64)
    list_of_inputs.append({
        'x': np.array([1, 2, 3, 0], dtype=np.int64),
        'name': 'perm_7'
    })
    
    # 8. Reversed 3-element permutation (int64)
    list_of_inputs.append({
        'x': np.array([2, 1, 0], dtype=np.int64),
        'name': 'perm_8'
    })
    
    # 9. Large elements (int32)
    list_of_inputs.append({
        'x': np.array([4, 0, 1, 2, 3], dtype=np.int32),
        'name': 'perm_9'
    })
    
    # 10. Multi-swap permutation (int32)
    list_of_inputs.append({
        'x': np.array([1, 0, 3, 2], dtype=np.int32),
        'name': 'perm_10'
    })
    
    return list_of_inputs

generated_inputs["tf.math.invert_permutation"] = tf_math_invert_permutation_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_segment_min_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.5, 1.5, 2.5]], dtype=np.float32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    name = None
    list_of_inputs.append({"data": data, "segment_ids": segment_ids, "name": name})

    # Input 2: 1D int32 array, with name
    data = np.array([10, -5, 20, 30, -100], dtype=np.int32)
    segment_ids = np.array([0, 0, 0, 1, 1], dtype=np.int32)
    name = "segment_min_1d"
    list_of_inputs.append({"data": data, "segment_ids": segment_ids, "name": name})

    # Input 3: 3D float64 array
    data = np.random.randn(4, 2, 2).astype(np.float64)
    segment_ids = np.array([0, 1, 1, 2], dtype=np.int64)
    name = None
    list_of_inputs.append({"data": data, "segment_ids": segment_ids, "name": name})

    # Input 4: 2D uint8 array
    data = np.array([[10, 20], [30, 40], [5, 15], [25, 35]], dtype=np.uint8)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    name = "uint8_test"
    list_of_inputs.append({"data": data, "segment_ids": segment_ids, "name": name})

    # Input 5: 2D int16 array with negative values
    data = np.array([[-10, 20], [-30, -40], [50, -15], [25, -35], [0, 0]], dtype=np.int16)
    segment_ids = np.array([0, 0, 1, 1, 1], dtype=np.int32)
    name = None
    list_of_inputs.append({"data": data, "segment_ids": segment_ids, "name": name})

    # Input 6: 1D float32 array with negative values
    data = np.array([-1.5, -2.5, 3.0, 4.2, -5.5], dtype=np.float32)
    segment_ids = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    name = "neg_float"
    list_of_inputs.append({"data": data, "segment_ids": segment_ids, "name": name})

    # Input 7: 4D float32 array
    data = np.random.randn(3, 2, 2, 2).astype(np.float32)
    segment_ids = np.array([0, 0, 0], dtype=np.int64)
    name = None
    list_of_inputs.append({"data": data, "segment_ids": segment_ids, "name": name})

    # Input 8: 2D int64 array with gaps in segment_ids
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int64)
    segment_ids = np.array([1, 1, 3, 3], dtype=np.int32)
    name = "gaps_in_ids"
    list_of_inputs.append({"data": data, "segment_ids": segment_ids, "name": name})

    # Input 9: 2D int8 array
    data = np.array([[1, 2, 3, 4, 5], [-1, -2, -3, -4, -5]], dtype=np.int8)
    segment_ids = np.array([0, 1], dtype=np.int32)
    name = None
    list_of_inputs.append({"data": data, "segment_ids": segment_ids, "name": name})

    # Input 10: 3D float64 array, single elements in segments
    data = np.array([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]], dtype=np.float64)
    segment_ids = np.array([0, 1, 2], dtype=np.int32)
    name = "single_element_segments"
    list_of_inputs.append({"data": data, "segment_ids": segment_ids, "name": name})

    return list_of_inputs

generated_inputs["tf.math.segment_min"] = tf_math_segment_min_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_segment_prod_inputs():
    list_of_inputs = []

    # Input 1: 1D Float32 array, 4 elements, 2 segments
    list_of_inputs.append({
        "data": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "segment_ids": np.array([0, 0, 1, 1], dtype=np.int32),
        "name": "segment_prod_1d_float32"
    })

    # Input 2: 2D Int32 array, shape (3, 2), 2 segments
    list_of_inputs.append({
        "data": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32),
        "segment_ids": np.array([0, 0, 1], dtype=np.int32),
        "name": "segment_prod_2d_int32"
    })

    # Input 3: 3D Float64 array, shape (4, 2, 2), 3 segments
    list_of_inputs.append({
        "data": np.array([
            [[1.0, 2.0], [3.0, 4.0]],
            [[-1.0, -2.0], [-3.0, -4.0]],
            [[0.5, 1.5], [2.5, 3.5]],
            [[2.0, 2.0], [2.0, 2.0]]
        ], dtype=np.float64),
        "segment_ids": np.array([0, 1, 1, 2], dtype=np.int64),
        "name": "segment_prod_3d_float64"
    })

    # Input 4: 1D Complex64 array, 3 elements, 1 segment
    list_of_inputs.append({
        "data": np.array([1+1j, 2-1j, 3+2j], dtype=np.complex64),
        "segment_ids": np.array([0, 0, 0], dtype=np.int32),
        "name": "segment_prod_1d_complex"
    })

    # Input 5: 2D Uint8 array, shape (2, 2), 2 segments (one element each)
    list_of_inputs.append({
        "data": np.array([[5, 10], [15, 20]], dtype=np.uint8),
        "segment_ids": np.array([0, 1], dtype=np.int32),
        "name": "segment_prod_2d_uint8"
    })

    # Input 6: 1D Int64 array with negative values, 5 elements, 2 segments
    list_of_inputs.append({
        "data": np.array([-1, -2, 3, 4, -5], dtype=np.int64),
        "segment_ids": np.array([0, 0, 1, 1, 1], dtype=np.int64),
        "name": "segment_prod_1d_int64_neg"
    })

    # Input 7: 4D Float32 array, shape (3, 2, 2, 2), 1 segment
    list_of_inputs.append({
        "data": np.ones((3, 2, 2, 2), dtype=np.float32) * 2.0,
        "segment_ids": np.array([0, 0, 0], dtype=np.int32),
        "name": "segment_prod_4d_float32"
    })

    # Input 8: 2D Float32 array with single element in dimension 0, 1 segment
    list_of_inputs.append({
        "data": np.array([[1.5, 2.5]], dtype=np.float32),
        "segment_ids": np.array([0], dtype=np.int32),
        "name": "segment_prod_single_element_dim0"
    })

    # Input 9: 1D Int16 array, 3 elements, 3 segments (each element in its own segment)
    list_of_inputs.append({
        "data": np.array([10, 20, 30], dtype=np.int16),
        "segment_ids": np.array([0, 1, 2], dtype=np.int32),
        "name": "segment_prod_1d_int16"
    })

    # Input 10: 3D Float32 array, shape (5, 1, 3), 3 segments
    list_of_inputs.append({
        "data": np.array([
            [[-1.0, 2.0, -3.0]],
            [[4.0, -5.0, 6.0]],
            [[-7.0, 8.0, -9.0]],
            [[1.0, 1.0, 1.0]],
            [[2.0, 3.0, 4.0]]
        ], dtype=np.float32),
        "segment_ids": np.array([0, 0, 1, 2, 2], dtype=np.int32),
        "name": "segment_prod_3d_with_gaps_float32"
    })

    return list_of_inputs

generated_inputs["tf.math.segment_prod"] = tf_math_segment_prod_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()

def tf_math_unsorted_segment_max_inputs():
    list_of_inputs = []

    # Case 1: Simple 1D float32
    list_of_inputs.append({
        "data": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "segment_ids": np.array([0, 0, 1, 1], dtype=np.int32),
        "num_segments": 2,
        "name": "segment_max_1d_float"
    })

    # Case 2: 2D data (3x4), 1D segment_ids (3)
    list_of_inputs.append({
        "data": np.array([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]], dtype=np.int32),
        "segment_ids": np.array([0, 1, 0], dtype=np.int32),
        "num_segments": 2,
        "name": "segment_max_2d_int"
    })

    # Case 3: 3D data (2x2x3), 1D segment_ids (2)
    list_of_inputs.append({
        "data": np.array([[[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], [[7.5, 8.5, 9.5], [10.5, 11.5, 12.5]]], dtype=np.float64),
        "segment_ids": np.array([1, 0], dtype=np.int32),
        "num_segments": 2,
        "name": "segment_max_3d_float"
    })

    # Case 4: 3D data (2x2x2), 2D segment_ids (2x2)
    list_of_inputs.append({
        "data": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32),
        "segment_ids": np.array([[0, 1], [0, 1]], dtype=np.int32),
        "num_segments": 2,
        "name": "segment_max_3d_2d_ids"
    })

    # Case 5: 2D data with negative values and some segment_ids negative (should be ignored)
    list_of_inputs.append({
        "data": np.array([[-1.0, -2.0], [3.0, 4.0], [-5.0, -6.0]], dtype=np.float32),
        "segment_ids": np.array([-1, 0, -1], dtype=np.int32),
        "num_segments": 1,
        "name": "segment_max_neg_ids"
    })

    # Case 6: 1D data with int64 and int64 segment_ids
    list_of_inputs.append({
        "data": np.array([10, 20, 30, 40, 50], dtype=np.int64),
        "segment_ids": np.array([0, 1, 2, 1, 0], dtype=np.int64),
        "num_segments": 3,
        "name": "segment_max_int64"
    })

    # Case 7: 2D data with uint8
    list_of_inputs.append({
        "data": np.array([[10, 20], [30, 40], [50, 60]], dtype=np.uint8),
        "segment_ids": np.array([0, 0, 1], dtype=np.int32),
        "num_segments": 2,
        "name": "segment_max_uint8"
    })

    # Case 8: 4D data (2x2x2x2), 2D segment_ids (2x2)
    list_of_inputs.append({
        "data": np.ones((2, 2, 2, 2), dtype=np.float32),
        "segment_ids": np.array([[0, 1], [1, 0]], dtype=np.int32),
        "num_segments": 3,
        "name": "segment_max_4d"
    })

    # Case 9: 1D data with empty segment (num_segments larger than max segment_id)
    list_of_inputs.append({
        "data": np.array([1.5, 2.5], dtype=np.float32),
        "segment_ids": np.array([0, 0], dtype=np.int32),
        "num_segments": 3,
        "name": "segment_max_empty_seg"
    })

    # Case 10: 3D data, 3D segment_ids (fully specified segments)
    list_of_inputs.append({
        "data": np.array([[[1], [2]], [[3], [4]]], dtype=np.float32),
        "segment_ids": np.array([[[1], [0]], [[0], [1]]], dtype=np.int32),
        "num_segments": 2,
        "name": "segment_max_full_ids"
    })

    return list_of_inputs

generated_inputs["tf.math.unsorted_segment_max"] = tf_math_unsorted_segment_max_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_unsorted_segment_mean_inputs():
    list_of_inputs = []
    
    # 1. Simple 1D float32
    list_of_inputs.append({
        'data': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        'segment_ids': np.array([0, 1, 0, 1], dtype=np.int32),
        'num_segments': 2,
        'name': 'simple_1d'
    })
    
    # 2. 2D data with 1D segment_ids
    list_of_inputs.append({
        'data': np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        'segment_ids': np.array([0, 1, 0], dtype=np.int32),
        'num_segments': 2,
        'name': '2d_data_1d_seg'
    })
    
    # 3. Negative segment ids (ignored)
    list_of_inputs.append({
        'data': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        'segment_ids': np.array([-1, 0, -1, 1], dtype=np.int32),
        'num_segments': 2,
        'name': 'neg_seg_ids'
    })
    
    # 4. float16 dtype
    list_of_inputs.append({
        'data': np.array([1.0, 3.0], dtype=np.float16),
        'segment_ids': np.array([0, 0], dtype=np.int32),
        'num_segments': 1,
        'name': 'float16_data'
    })
    
    # 5. Higher dimensional data with multi-dimensional segment_ids
    list_of_inputs.append({
        'data': np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        'segment_ids': np.array([[0, 1], [2, 0]], dtype=np.int32),
        'num_segments': 3,
        'name': 'high_dim'
    })
    
    # 6. Unused segments (output will have 0s)
    list_of_inputs.append({
        'data': np.array([10.0, 20.0], dtype=np.float32),
        'segment_ids': np.array([0, 0], dtype=np.int32),
        'num_segments': 3,
        'name': 'unused_segments'
    })
    
    # 7. All negative segment ids
    list_of_inputs.append({
        'data': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'segment_ids': np.array([-1, -1, -2], dtype=np.int32),
        'num_segments': 2,
        'name': 'all_neg_seg_ids'
    })
    
    # 8. 3D data, 1D segment_ids
    list_of_inputs.append({
        'data': np.ones((3, 2, 2), dtype=np.float32),
        'segment_ids': np.array([0, 1, 1], dtype=np.int32),
        'num_segments': 2,
        'name': '3d_data_1d_seg'
    })
    
    # 9. 4D data, 3D segment_ids
    list_of_inputs.append({
        'data': np.ones((2, 2, 2, 2), dtype=np.float64),
        'segment_ids': np.array([[[0, 1], [1, 0]], [[0, 0], [1, 1]]], dtype=np.int32),
        'num_segments': 2,
        'name': '4d_data_3d_seg'
    })
    
    # 10. Single element data, large num_segments
    list_of_inputs.append({
        'data': np.array([1.5], dtype=np.float32),
        'segment_ids': np.array([4], dtype=np.int32),
        'num_segments': 5,
        'name': 'single_elem'
    })
    
    return list_of_inputs

generated_inputs["tf.math.unsorted_segment_mean"] = tf_math_unsorted_segment_mean_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_unsorted_segment_sum_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 array
    input_dict = {
        "data": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "segment_ids": np.array([0, 1, 0, 2, 1], dtype=np.int32),
        "num_segments": 3,
        "name": "basic_1d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array with standard segments
    input_dict = {
        "data": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32),
        "segment_ids": np.array([0, 1, 0], dtype=np.int32),
        "num_segments": 2,
        "name": "basic_2d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Using negative segment IDs (values are ignored)
    input_dict = {
        "data": np.array([10, 20, 30, 40], dtype=np.int64),
        "segment_ids": np.array([-1, 0, -1, 1], dtype=np.int64),
        "num_segments": 2,
        "name": "negative_indices"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float64 data with 1D segment IDs
    input_dict = {
        "data": np.ones((2, 3, 2), dtype=np.float64),
        "segment_ids": np.array([0, 1], dtype=np.int32),
        "num_segments": 3,
        "name": "3d_data"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float64 data, int64 segment IDs, and unused segments
    input_dict = {
        "data": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        "segment_ids": np.array([1, 1, 0], dtype=np.int64),
        "num_segments": 4,
        "name": "unused_segments"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single-element float32 array
    input_dict = {
        "data": np.array([1.5], dtype=np.float32),
        "segment_ids": np.array([0], dtype=np.int32),
        "num_segments": 1,
        "name": "single_element"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Out-of-order segment IDs
    input_dict = {
        "data": np.array([1, 2, 3, 4, 5], dtype=np.int16),
        "segment_ids": np.array([4, 2, 0, 1, 3], dtype=np.int32),
        "num_segments": 5,
        "name": "out_of_order"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex64 data types
    input_dict = {
        "data": np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64),
        "segment_ids": np.array([0, 1, 0], dtype=np.int32),
        "num_segments": 2,
        "name": "complex_data"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Multi-dimensional segment IDs (prefix of shape)
    input_dict = {
        "data": np.ones((2, 2, 3), dtype=np.float32),
        "segment_ids": np.array([[0, 1], [1, 0]], dtype=np.int32),
        "num_segments": 2,
        "name": "multi_dim_ids"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Sparse segment IDs with a large num_segments
    input_dict = {
        "data": np.array([5, 10], dtype=np.int32),
        "segment_ids": np.array([1, 3], dtype=np.int32),
        "num_segments": 5,
        "name": "sparse_segments"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.unsorted_segment_sum"] = tf_math_unsorted_segment_sum_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_atrous_conv2d_inputs():
    list_of_inputs = []
    
    # Input 1
    value = np.random.randn(1, 3, 3, 1).astype(np.float32)
    filters = np.random.randn(2, 2, 1, 1).astype(np.float32)
    rate = 1
    padding = 'VALID'
    name = 'atrous_conv_1'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 2
    value = np.random.randn(2, 5, 5, 3).astype(np.float32)
    filters = np.random.randn(3, 3, 3, 4).astype(np.float32)
    rate = 2
    padding = 'SAME'
    name = 'atrous_conv_2'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 3
    value = np.random.randn(1, 7, 7, 2).astype(np.float64)
    filters = np.random.randn(3, 3, 2, 2).astype(np.float64)
    rate = 3
    padding = 'VALID'
    name = 'atrous_conv_3'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 4
    value = np.random.randn(4, 10, 10, 8).astype(np.float32)
    filters = np.random.randn(1, 1, 8, 16).astype(np.float32)
    rate = 1
    padding = 'SAME'
    name = 'atrous_conv_4'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 5
    value = np.random.randn(2, 8, 12, 3).astype(np.float32)
    filters = np.random.randn(3, 5, 3, 4).astype(np.float32)
    rate = 2
    padding = 'VALID'
    name = 'atrous_conv_5'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 6
    value = np.random.randn(1, 14, 14, 1).astype(np.float32)
    filters = np.random.randn(5, 5, 1, 2).astype(np.float32)
    rate = 4
    padding = 'SAME'
    name = 'atrous_conv_6'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 7
    value = np.random.randn(1, 20, 20, 4).astype(np.float32)
    filters = np.random.randn(2, 2, 4, 4).astype(np.float32)
    rate = 5
    padding = 'SAME'
    name = 'atrous_conv_7'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 8
    value = np.random.randn(3, 6, 6, 16).astype(np.float32)
    filters = np.random.randn(1, 1, 16, 8).astype(np.float32)
    rate = 2
    padding = 'VALID'
    name = 'atrous_conv_8'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 9
    value = np.random.randn(1, 5, 5, 32).astype(np.float32)
    filters = np.random.randn(3, 3, 32, 64).astype(np.float32)
    rate = 1
    padding = 'SAME'
    name = 'atrous_conv_9'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 10
    value = np.random.randn(2, 15, 15, 3).astype(np.float32)
    filters = np.random.randn(3, 3, 3, 3).astype(np.float32)
    rate = 3
    padding = 'VALID'
    name = 'atrous_conv_10'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    return list_of_inputs

generated_inputs["tf.nn.atrous_conv2d"] = tf_nn_atrous_conv2d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_avg_pool_inputs():
    list_of_inputs = []
    
    # Input 1: N=2, NHWC, 2x2 pool, 2x2 stride
    input_1 = np.random.randn(2, 8, 8, 3).astype(np.float32)
    ksize_1 = [2, 2]
    strides_1 = [2, 2]
    padding_1 = "VALID"
    data_format_1 = "NHWC"
    name_1 = "pool1"
    
    list_of_inputs.append(copy.deepcopy({
        "input": input_1,
        "ksize": ksize_1,
        "strides": strides_1,
        "padding": padding_1,
        "data_format": data_format_1,
        "name": name_1
    }))
    
    # Input 2: N=2, NHWC, SAME padding, length N+2 ksize
    input_2 = np.random.randn(2, 8, 8, 3).astype(np.float32)
    ksize_2 = [1, 2, 2, 1]
    strides_2 = [1, 2, 2, 1]
    padding_2 = "SAME"
    data_format_2 = "NHWC"
    name_2 = "pool2"
    
    list_of_inputs.append(copy.deepcopy({
        "input": input_2,
        "ksize": ksize_2,
        "strides": strides_2,
        "padding": padding_2,
        "data_format": data_format_2,
        "name": name_2
    }))
    
    # Input 3: N=1, NWC, 2 pool, 1 stride
    input_3 = np.random.randn(1, 10, 4).astype(np.float32)
    ksize_3 = [2]
    strides_3 = [1]
    padding_3 = "VALID"
    data_format_3 = "NWC"
    name_3 = "pool3"
    
    list_of_inputs.append(copy.deepcopy({
        "input": input_3,
        "ksize": ksize_3,
        "strides": strides_3,
        "padding": padding_3,
        "data_format": data_format_3,
        "name": name_3
    }))

    # Input 4: N=1, NWC, SAME padding, length N+2 ksize
    input_4 = np.random.randn(1, 10, 4).astype(np.float32)
    ksize_4 = [1, 2, 1]
    strides_4 = [1, 1, 1]
    padding_4 = "SAME"
    data_format_4 = "NWC"
    name_4 = "pool4"
    
    list_of_inputs.append(copy.deepcopy({
        "input": input_4,
        "ksize": ksize_4,
        "strides": strides_4,
        "padding": padding_4,
        "data_format": data_format_4,
        "name": name_4
    }))

    # Input 5: N=3, NDHWC, 2x2x2 pool, 2x2x2 stride
    input_5 = np.random.randn(2, 4, 4, 4, 3).astype(np.float32)
    ksize_5 = [2, 2, 2]
    strides_5 = [2, 2, 2]
    padding_5 = "VALID"
    data_format_5 = "NDHWC"
    name_5 = "pool5"
    
    list_of_inputs.append(copy.deepcopy({
        "input": input_5,
        "ksize": ksize_5,
        "strides": strides_5,
        "padding": padding_5,
        "data_format": data_format_5,
        "name": name_5
    }))

    # Input 6: N=3, NDHWC, SAME padding, length N+2 ksize
    input_6 = np.random.randn(2, 4, 4, 4, 3).astype(np.float32)
    ksize_6 = [1, 2, 2, 2, 1]
    strides_6 = [1, 2, 2, 2, 1]
    padding_6 = "SAME"
    data_format_6 = "NDHWC"
    name_6 = "pool6"
    
    list_of_inputs.append(copy.deepcopy({
        "input": input_6,
        "ksize": ksize_6,
        "strides": strides_6,
        "padding": padding_6,
        "data_format": data_format_6,
        "name": name_6
    }))

    # Input 7: N=2, NHWC, float64
    input_7 = np.random.randn(1, 6, 6, 2).astype(np.float64)
    ksize_7 = [3, 3]
    strides_7 = [1, 1]
    padding_7 = "VALID"
    data_format_7 = "NHWC"
    name_7 = "pool7"
    
    list_of_inputs.append(copy.deepcopy({
        "input": input_7,
        "ksize": ksize_7,
        "strides": strides_7,
        "padding": padding_7,
        "data_format": data_format_7,
        "name": name_7
    }))

    # Input 8: N=2, NHWC, length 1 ksize
    input_8 = np.random.uniform(-10.0, 10.0, (2, 5, 5, 2)).astype(np.float32)
    ksize_8 = [1]
    strides_8 = [1]
    padding_8 = "SAME"
    data_format_8 = "NHWC"
    name_8 = "pool8"
    
    list_of_inputs.append(copy.deepcopy({
        "input": input_8,
        "ksize": ksize_8,
        "strides": strides_8,
        "padding": padding_8,
        "data_format": data_format_8,
        "name": name_8
    }))

    # Input 9: N=2, NHWC, stride larger than pool size
    input_9 = np.random.randn(1, 10, 10, 2).astype(np.float32)
    ksize_9 = [2, 2]
    strides_9 = [3, 3]
    padding_9 = "VALID"
    data_format_9 = "NHWC"
    name_9 = "pool9"
    
    list_of_inputs.append(copy.deepcopy({
        "input": input_9,
        "ksize": ksize_9,
        "strides": strides_9,
        "padding": padding_9,
        "data_format": data_format_9,
        "name": name_9
    }))

    # Input 10: N=1, NWC, large strides
    input_10 = np.random.randn(4, 20, 1).astype(np.float32)
    ksize_10 = [5]
    strides_10 = [5]
    padding_10 = "SAME"
    data_format_10 = "NWC"
    name_10 = "pool10"
    
    list_of_inputs.append(copy.deepcopy({
        "input": input_10,
        "ksize": ksize_10,
        "strides": strides_10,
        "padding": padding_10,
        "data_format": data_format_10,
        "name": name_10
    }))

    return list_of_inputs

generated_inputs["tf.nn.avg_pool"] = tf_nn_avg_pool_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_avg_pool1d_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "input": np.random.randn(2, 10, 3).astype(np.float32),
        "ksize": [2],
        "strides": [2],
        "padding": "VALID",
        "data_format": "NWC",
        "name": "pool_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "input": np.ones((2, 8, 3), dtype=np.float32),
        "ksize": [3],
        "strides": [2],
        "padding": "SAME",
        "data_format": "NWC",
        "name": "pool_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "input": np.zeros((1, 10, 4), dtype=np.float32),
        "ksize": [1, 2, 1],
        "strides": [1, 1, 1],
        "padding": "VALID",
        "data_format": "NWC",
        "name": "pool_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "input": np.random.uniform(-5.0, 5.0, (3, 6, 4)).astype(np.float32),
        "ksize": [1, 3, 1],
        "strides": [1, 2, 1],
        "padding": "SAME",
        "data_format": "NWC",
        "name": "pool_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "input": np.arange(30, dtype=np.float32).reshape(1, 10, 3),
        "ksize": [4],
        "strides": [3],
        "padding": "VALID",
        "data_format": "NWC",
        "name": "pool_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "input": np.random.randn(2, 12, 4).astype(np.float64),
        "ksize": [2],
        "strides": [1],
        "padding": "SAME",
        "data_format": "NWC",
        "name": "pool_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "input": np.ones((1, 15, 1), dtype=np.float32) * -2.5,
        "ksize": [1, 5, 1],
        "strides": [1, 5, 1],
        "padding": "VALID",
        "data_format": "NWC",
        "name": "pool_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "input": np.random.randn(4, 16, 3).astype(np.float32),
        "ksize": [1, 2, 1],
        "strides": [1, 2, 1],
        "padding": "SAME",
        "data_format": "NWC",
        "name": "pool_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "input": np.arange(24, dtype=np.float32).reshape(2, 4, 3),
        "ksize": [3],
        "strides": [1],
        "padding": "SAME",
        "data_format": "NWC",
        "name": "pool_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "input": np.random.normal(size=(1, 5, 5)).astype(np.float32),
        "ksize": [3],
        "strides": [2],
        "padding": "VALID",
        "data_format": "NWC",
        "name": "pool_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.avg_pool1d"] = tf_nn_avg_pool1d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_avg_pool1d_inputs():
    list_of_inputs = []
    
    # Input 1
    list_of_inputs.append({
        'input': np.random.randn(2, 10, 3).astype(np.float32),
        'ksize': 2,
        'strides': 2,
        'padding': 'VALID',
        'data_format': 'NWC',
        'name': 'pool1'
    })
    
    # Input 2
    list_of_inputs.append({
        'input': np.random.randn(1, 16, 4).astype(np.float32),
        'ksize': 3,
        'strides': 1,
        'padding': 'SAME',
        'data_format': 'NWC',
        'name': 'pool2'
    })
    
    # Input 3
    list_of_inputs.append({
        'input': np.random.randn(4, 8, 2).astype(np.float32),
        'ksize': 2,
        'strides': 1,
        'padding': 'VALID',
        'data_format': 'NWC',
        'name': 'pool3'
    })
    
    # Input 4
    list_of_inputs.append({
        'input': np.random.randn(3, 20, 3).astype(np.float64),
        'ksize': 4,
        'strides': 2,
        'padding': 'SAME',
        'data_format': 'NWC',
        'name': 'pool4'
    })
    
    # Input 5
    list_of_inputs.append({
        'input': np.arange(24).reshape(2, 4, 3).astype(np.float32),
        'ksize': 2,
        'strides': 2,
        'padding': 'VALID',
        'data_format': 'NWC',
        'name': 'pool5'
    })
    
    # Input 6
    list_of_inputs.append({
        'input': np.ones((1, 100, 1), dtype=np.float32),
        'ksize': 5,
        'strides': 5,
        'padding': 'SAME',
        'data_format': 'NWC',
        'name': 'pool6'
    })
    
    # Input 7
    list_of_inputs.append({
        'input': np.zeros((2, 10, 5), dtype=np.float32),
        'ksize': 3,
        'strides': 3,
        'padding': 'VALID',
        'data_format': 'NWC',
        'name': 'pool7'
    })
    
    # Input 8
    list_of_inputs.append({
        'input': np.random.uniform(-10, 10, (5, 8, 2)).astype(np.float32),
        'ksize': 1,
        'strides': 1,
        'padding': 'VALID',
        'data_format': 'NWC',
        'name': 'pool8'
    })
    
    # Input 9
    list_of_inputs.append({
        'input': np.random.normal(0, 1, (2, 12, 2)).astype(np.float64),
        'ksize': 3,
        'strides': 2,
        'padding': 'SAME',
        'data_format': 'NWC',
        'name': 'pool9'
    })
    
    # Input 10
    list_of_inputs.append({
        'input': np.ones((4, 30, 3), dtype=np.float32),
        'ksize': 10,
        'strides': 5,
        'padding': 'SAME',
        'data_format': 'NWC',
        'name': 'pool10'
    })
    
    return list_of_inputs

generated_inputs["tf.nn.avg_pool1d_1"] = tf_nn_avg_pool1d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_avg_pool2d_inputs():
    list_of_inputs = []

    # Input 1
    input_1 = np.random.randn(1, 4, 4, 1).astype(np.float32)
    ksize_1 = [2, 2]
    strides_1 = [2, 2]
    padding_1 = "VALID"
    data_format_1 = "NHWC"
    name_1 = "avg_pool_1"
    list_of_inputs.append({
        "input": input_1,
        "ksize": ksize_1,
        "strides": strides_1,
        "padding": padding_1,
        "data_format": data_format_1,
        "name": name_1
    })

    # Input 2
    input_2 = np.random.randn(2, 8, 8, 3).astype(np.float32)
    ksize_2 = [1, 2, 2, 1]
    strides_2 = [1, 2, 2, 1]
    padding_2 = "SAME"
    data_format_2 = "NHWC"
    name_2 = "avg_pool_2"
    list_of_inputs.append({
        "input": input_2,
        "ksize": ksize_2,
        "strides": strides_2,
        "padding": padding_2,
        "data_format": data_format_2,
        "name": name_2
    })

    # Input 3
    input_3 = np.random.randn(1, 10, 10, 2).astype(np.float64)
    ksize_3 = [1, 3, 3, 1]
    strides_3 = [1, 2, 2, 1]
    padding_3 = "VALID"
    data_format_3 = "NHWC"
    name_3 = "avg_pool_3"
    list_of_inputs.append({
        "input": input_3,
        "ksize": ksize_3,
        "strides": strides_3,
        "padding": padding_3,
        "data_format": data_format_3,
        "name": name_3
    })

    # Input 4
    input_4 = np.random.randn(4, 16, 16, 64).astype(np.float32)
    ksize_4 = [4, 4]
    strides_4 = [4, 4]
    padding_4 = "SAME"
    data_format_4 = "NHWC"
    name_4 = "avg_pool_4"
    list_of_inputs.append({
        "input": input_4,
        "ksize": ksize_4,
        "strides": strides_4,
        "padding": padding_4,
        "data_format": data_format_4,
        "name": name_4
    })

    # Input 5
    input_5 = np.random.randn(1, 5, 5, 3).astype(np.float32)
    ksize_5 = [2, 2]
    strides_5 = [1, 1]
    padding_5 = "SAME"
    data_format_5 = "NHWC"
    name_5 = "avg_pool_5"
    list_of_inputs.append({
        "input": input_5,
        "ksize": ksize_5,
        "strides": strides_5,
        "padding": padding_5,
        "data_format": data_format_5,
        "name": name_5
    })

    # Input 6
    input_6 = np.random.randn(2, 14, 14, 3).astype(np.float32)
    ksize_6 = [2, 2]
    strides_6 = [2, 2]
    padding_6 = "VALID"
    data_format_6 = "NHWC"
    name_6 = "avg_pool_6"
    list_of_inputs.append({
        "input": input_6,
        "ksize": ksize_6,
        "strides": strides_6,
        "padding": padding_6,
        "data_format": data_format_6,
        "name": name_6
    })

    # Input 7
    input_7 = np.random.randn(1, 1, 1, 1).astype(np.float32)
    ksize_7 = [1, 1]
    strides_7 = [1, 1]
    padding_7 = "VALID"
    data_format_7 = "NHWC"
    name_7 = "avg_pool_7"
    list_of_inputs.append({
        "input": input_7,
        "ksize": ksize_7,
        "strides": strides_7,
        "padding": padding_7,
        "data_format": data_format_7,
        "name": name_7
    })

    # Input 8
    input_8 = np.random.randn(8, 32, 32, 3).astype(np.float32)
    ksize_8 = [2, 2]
    strides_8 = [1, 1]
    padding_8 = "SAME"
    data_format_8 = "NHWC"
    name_8 = "avg_pool_8"
    list_of_inputs.append({
        "input": input_8,
        "ksize": ksize_8,
        "strides": strides_8,
        "padding": padding_8,
        "data_format": data_format_8,
        "name": name_8
    })

    # Input 9
    input_9 = np.random.randn(2, 14, 14, 128).astype(np.float64)
    ksize_9 = [1, 3, 3, 1]
    strides_9 = [1, 1, 1, 1]
    padding_9 = "VALID"
    data_format_9 = "NHWC"
    name_9 = "avg_pool_9"
    list_of_inputs.append({
        "input": input_9,
        "ksize": ksize_9,
        "strides": strides_9,
        "padding": padding_9,
        "data_format": data_format_9,
        "name": name_9
    })

    # Input 10
    input_10 = np.random.randn(3, 28, 28, 16).astype(np.float32)
    ksize_10 = [1, 2, 2, 1]
    strides_10 = [1, 2, 2, 1]
    padding_10 = "SAME"
    data_format_10 = "NHWC"
    name_10 = "avg_pool_10"
    list_of_inputs.append({
        "input": input_10,
        "ksize": ksize_10,
        "strides": strides_10,
        "padding": padding_10,
        "data_format": data_format_10,
        "name": name_10
    })

    return list_of_inputs

generated_inputs["tf.nn.avg_pool2d"] = tf_nn_avg_pool2d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_avg_pool3d_inputs():
    list_of_inputs = []

    # Input 1
    input_1 = np.random.randn(2, 3, 4, 4, 3).astype(np.float32)
    ksize_1 = [1, 2, 2, 2, 1]
    strides_1 = [1, 1, 1, 1, 1]
    padding_1 = "VALID"
    data_format_1 = "NDHWC"
    name_1 = "pool_1"
    list_of_inputs.append({
        "input": input_1,
        "ksize": ksize_1,
        "strides": strides_1,
        "padding": padding_1,
        "data_format": data_format_1,
        "name": name_1
    })

    # Input 2
    input_2 = np.random.randn(1, 2, 8, 8, 1).astype(np.float32)
    ksize_2 = [1, 1, 2, 2, 1]
    strides_2 = [1, 1, 2, 2, 1]
    padding_2 = "SAME"
    data_format_2 = "NDHWC"
    name_2 = "pool_2"
    list_of_inputs.append({
        "input": input_2,
        "ksize": ksize_2,
        "strides": strides_2,
        "padding": padding_2,
        "data_format": data_format_2,
        "name": name_2
    })

    # Input 3
    input_3 = np.random.randn(2, 4, 4, 4, 3).astype(np.float32)
    ksize_3 = [1, 2, 2, 2, 1]
    strides_3 = [1, 1, 1, 1, 1]
    padding_3 = "VALID"
    data_format_3 = "NDHWC"
    name_3 = "pool_3"
    list_of_inputs.append({
        "input": input_3,
        "ksize": ksize_3,
        "strides": strides_3,
        "padding": padding_3,
        "data_format": data_format_3,
        "name": name_3
    })

    # Input 4
    input_4 = np.random.randn(2, 4, 4, 4, 2).astype(np.float32)
    ksize_4 = [2, 2, 2]
    strides_4 = [1, 1, 1]
    padding_4 = "SAME"
    data_format_4 = "NDHWC"
    name_4 = "pool_4"
    list_of_inputs.append({
        "input": input_4,
        "ksize": ksize_4,
        "strides": strides_4,
        "padding": padding_4,
        "data_format": data_format_4,
        "name": name_4
    })

    # Input 5
    input_5 = np.random.randn(1, 10, 10, 10, 1).astype(np.float32)
    ksize_5 = [3, 3, 3]
    strides_5 = [2, 2, 2]
    padding_5 = "VALID"
    data_format_5 = "NDHWC"
    name_5 = "pool_5"
    list_of_inputs.append({
        "input": input_5,
        "ksize": ksize_5,
        "strides": strides_5,
        "padding": padding_5,
        "data_format": data_format_5,
        "name": name_5
    })

    # Input 6
    input_6 = np.random.randn(1, 5, 5, 5, 2).astype(np.float32)
    ksize_6 = [2, 2, 2]
    strides_6 = [2, 2, 2]
    padding_6 = "SAME"
    data_format_6 = "NDHWC"
    name_6 = "pool_6"
    list_of_inputs.append({
        "input": input_6,
        "ksize": ksize_6,
        "strides": strides_6,
        "padding": padding_6,
        "data_format": data_format_6,
        "name": name_6
    })

    # Input 7
    input_7 = np.random.randn(2, 3, 3, 3, 2).astype(np.float32)
    ksize_7 = [1]
    strides_7 = [1]
    padding_7 = "VALID"
    data_format_7 = "NDHWC"
    name_7 = "pool_7"
    list_of_inputs.append({
        "input": input_7,
        "ksize": ksize_7,
        "strides": strides_7,
        "padding": padding_7,
        "data_format": data_format_7,
        "name": name_7
    })

    # Input 8
    input_8 = np.random.randn(2, 3, 3, 3, 1).astype(np.float32)
    ksize_8 = [1]
    strides_8 = [1]
    padding_8 = "SAME"
    data_format_8 = "NDHWC"
    name_8 = "pool_8"
    list_of_inputs.append({
        "input": input_8,
        "ksize": ksize_8,
        "strides": strides_8,
        "padding": padding_8,
        "data_format": data_format_8,
        "name": name_8
    })

    # Input 9
    input_9 = np.random.randn(1, 6, 6, 6, 2).astype(np.float32)
    ksize_9 = [1, 3, 3, 3, 1]
    strides_9 = [1, 3, 3, 3, 1]
    padding_9 = "VALID"
    data_format_9 = "NDHWC"
    name_9 = "pool_9"
    list_of_inputs.append({
        "input": input_9,
        "ksize": ksize_9,
        "strides": strides_9,
        "padding": padding_9,
        "data_format": data_format_9,
        "name": name_9
    })

    # Input 10
    input_10 = np.random.randn(4, 2, 2, 2, 3).astype(np.float32)
    ksize_10 = [1, 1, 1, 1, 1]
    strides_10 = [1, 1, 1, 1, 1]
    padding_10 = "SAME"
    data_format_10 = "NDHWC"
    name_10 = "pool_10"
    list_of_inputs.append({
        "input": input_10,
        "ksize": ksize_10,
        "strides": strides_10,
        "padding": padding_10,
        "data_format": data_format_10,
        "name": name_10
    })

    return list_of_inputs

generated_inputs["tf.nn.avg_pool3d"] = tf_nn_avg_pool3d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_bias_add_inputs():
    list_of_inputs = []

    # Input 1: float32, NHWC format, 2D array (N, C)
    value = np.random.randn(2, 3).astype(np.float32)
    bias = np.random.randn(3).astype(np.float32)
    input_dict = {
        'value': value,
        'bias': bias,
        'data_format': 'NHWC',
        'name': 'bias_add_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32, NHWC format, 2D array (N, C)
    value = np.random.randint(-10, 10, size=(3, 5)).astype(np.int32)
    bias = np.random.randint(-5, 5, size=(5,)).astype(np.int32)
    input_dict = {
        'value': value,
        'bias': bias,
        'data_format': 'NHWC',
        'name': 'bias_add_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, NHWC format, 3D array (N, H, C)
    value = np.random.randn(2, 4, 3).astype(np.float64)
    bias = np.random.randn(3).astype(np.float64)
    input_dict = {
        'value': value,
        'bias': bias,
        'data_format': 'NHWC',
        'name': 'bias_add_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32, NCHW format, 3D array (N, C, H)
    value = np.random.randn(2, 3, 4).astype(np.float32)
    bias = np.random.randn(3).astype(np.float32)
    input_dict = {
        'value': value,
        'bias': bias,
        'data_format': 'NCHW',
        'name': 'bias_add_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, NHWC format, 4D array (N, H, W, C)
    value = np.random.randn(2, 5, 5, 4).astype(np.float32)
    bias = np.random.randn(4).astype(np.float32)
    input_dict = {
        'value': value,
        'bias': bias,
        'data_format': 'NHWC',
        'name': 'bias_add_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float32, NCHW format, 4D array (N, C, H, W)
    value = np.random.randn(2, 4, 5, 5).astype(np.float32)
    bias = np.random.randn(4).astype(np.float32)
    input_dict = {
        'value': value,
        'bias': bias,
        'data_format': 'NCHW',
        'name': 'bias_add_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int64, NHWC format, 2D array (N, C)
    value = np.random.randint(-100, 100, size=(10, 2)).astype(np.int64)
    bias = np.random.randint(-50, 50, size=(2,)).astype(np.int64)
    input_dict = {
        'value': value,
        'bias': bias,
        'data_format': 'NHWC',
        'name': 'bias_add_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, NCHW format, 4D array (N, C, H, W)
    value = np.random.randn(1, 3, 8, 8).astype(np.float64)
    bias = np.random.randn(3).astype(np.float64)
    input_dict = {
        'value': value,
        'bias': bias,
        'data_format': 'NCHW',
        'name': 'bias_add_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, NHWC format, 4D array with larger channels
    value = np.random.randn(4, 3, 3, 16).astype(np.float32)
    bias = np.random.randn(16).astype(np.float32)
    input_dict = {
        'value': value,
        'bias': bias,
        'data_format': 'NHWC',
        'name': 'bias_add_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, NCHW format, 4D array with larger channels
    value = np.random.randn(4, 16, 3, 3).astype(np.float32)
    bias = np.random.randn(16).astype(np.float32)
    input_dict = {
        'value': value,
        'bias': bias,
        'data_format': 'NCHW',
        'name': 'bias_add_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.bias_add"] = tf_nn_bias_add_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_collapse_repeated_inputs():
    list_of_inputs = []
    
    # Input 1
    labels = np.array([[1, 1, 2, 2, 1], [1, 2, 3, 3, 3]], dtype=np.int32)
    seq_length = np.array([5, 5], dtype=np.int32)
    name = "collapse_1"
    list_of_inputs.append({
        "labels": labels,
        "seq_length": seq_length,
        "name": name
    })
    
    # Input 2
    labels = np.array([[1, 2, 2, 3], [4, 4, 4, 4]], dtype=np.int64)
    seq_length = np.array([3, 4], dtype=np.int64)
    name = "collapse_2"
    list_of_inputs.append({
        "labels": labels,
        "seq_length": seq_length,
        "name": name
    })
    
    # Input 3
    labels = np.array([[9, 9, 9], [8, 8, 7], [1, 2, 3]], dtype=np.int32)
    seq_length = np.array([3, 2, 1], dtype=np.int32)
    name = "collapse_3"
    list_of_inputs.append({
        "labels": labels,
        "seq_length": seq_length,
        "name": name
    })
    
    # Input 4
    labels = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
    seq_length = np.array([5], dtype=np.int32)
    name = "collapse_4"
    list_of_inputs.append({
        "labels": labels,
        "seq_length": seq_length,
        "name": name
    })
    
    # Input 5
    labels = np.array([[1, 2, 1, 2, 1, 2]], dtype=np.int64)
    seq_length = np.array([6], dtype=np.int64)
    name = "collapse_5"
    list_of_inputs.append({
        "labels": labels,
        "seq_length": seq_length,
        "name": name
    })
    
    # Input 6
    labels = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.int32)
    seq_length = np.array([3, 3], dtype=np.int32)
    name = "collapse_6"
    list_of_inputs.append({
        "labels": labels,
        "seq_length": seq_length,
        "name": name
    })
    
    # Input 7
    labels = np.array([[10, 10, 20, 20, 30, 30, 40, 40]], dtype=np.int32)
    seq_length = np.array([8], dtype=np.int32)
    name = "collapse_7"
    list_of_inputs.append({
        "labels": labels,
        "seq_length": seq_length,
        "name": name
    })
    
    # Input 8
    labels = np.array([[-1, -1, -2, -2], [-3, -3, -3, -4]], dtype=np.int32)
    seq_length = np.array([4, 4], dtype=np.int32)
    name = "collapse_8"
    list_of_inputs.append({
        "labels": labels,
        "seq_length": seq_length,
        "name": name
    })
    
    # Input 9
    labels = np.array([[100, 100], [200, 200], [300, 300]], dtype=np.int64)
    seq_length = np.array([2, 1, 2], dtype=np.int64)
    name = "collapse_9"
    list_of_inputs.append({
        "labels": labels,
        "seq_length": seq_length,
        "name": name
    })
    
    # Input 10
    labels = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=np.int32)
    seq_length = np.array([10], dtype=np.int32)
    name = "collapse_10"
    list_of_inputs.append({
        "labels": labels,
        "seq_length": seq_length,
        "name": name
    })
    
    return list_of_inputs

generated_inputs["tf.nn.collapse_repeated"] = tf_nn_collapse_repeated_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_conv1d_inputs():
    list_of_inputs = []
    
    # Input 1
    input_dict = {
        'input': np.random.randn(2, 10, 3).astype(np.float32),
        'filters': np.random.randn(3, 3, 4).astype(np.float32),
        'stride': 1,
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'conv1d_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'input': np.random.randn(1, 15, 2).astype(np.float32),
        'filters': np.random.randn(2, 2, 2).astype(np.float32),
        'stride': 2,
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'conv1d_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'input': np.random.randn(3, 8, 4).astype(np.float32),
        'filters': np.random.randn(3, 4, 5).astype(np.float32),
        'stride': 1,
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'conv1d_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'input': np.random.randn(2, 20, 4).astype(np.float64),
        'filters': np.random.randn(5, 4, 2).astype(np.float64),
        'stride': 3,
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'conv1d_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'input': np.random.randn(4, 12, 3).astype(np.float16),
        'filters': np.random.randn(2, 3, 6).astype(np.float16),
        'stride': 1,
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [2],
        'name': 'conv1d_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'input': np.random.randn(1, 16, 4).astype(np.float32),
        'filters': np.random.randn(3, 4, 8).astype(np.float32),
        'stride': 1,
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'conv1d_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'input': np.random.randn(2, 5, 3).astype(np.float32),
        'filters': np.random.randn(1, 3, 2).astype(np.float32),
        'stride': 1,
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': [2],
        'name': 'conv1d_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'input': -np.random.rand(2, 10, 3).astype(np.float32),
        'filters': np.random.randn(3, 3, 3).astype(np.float32),
        'stride': 2,
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'conv1d_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'input': np.random.randn(5, 5, 5).astype(np.float64),
        'filters': -np.random.rand(2, 5, 2).astype(np.float64),
        'stride': 1,
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'conv1d_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'input': np.zeros((2, 8, 3), dtype=np.float32),
        'filters': np.ones((3, 3, 4), dtype=np.float32),
        'stride': 1,
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'conv1d_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.conv1d"] = tf_nn_conv1d_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_nn_conv1d_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'input': np.random.randn(2, 10, 3).astype(np.float32),
        'filters': np.random.randn(3, 3, 4).astype(np.float32),
        'stride': [1],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'conv1d_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'input': np.random.randn(2, 10, 3).astype(np.float32),
        'filters': np.random.randn(3, 3, 4).astype(np.float32),
        'stride': [1],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'conv1d_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'input': np.random.randn(1, 8, 2).astype(np.float16),
        'filters': np.random.randn(2, 2, 2).astype(np.float16),
        'stride': [2],
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'conv1d_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'input': np.random.randn(4, 20, 4).astype(np.float64),
        'filters': np.random.randn(5, 4, 8).astype(np.float64),
        'stride': [3],
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'conv1d_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'input': np.random.randn(3, 12, 4).astype(np.float32),
        'filters': np.random.randn(3, 4, 5).astype(np.float32),
        'stride': [1],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'conv1d_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'input': np.random.randn(2, 16, 3).astype(np.float32),
        'filters': np.random.randn(3, 3, 4).astype(np.float32),
        'stride': [1],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [2],
        'name': 'conv1d_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'input': np.random.randn(2, 10, 3).astype(np.float32),
        'filters': np.random.randn(3, 3, 4).astype(np.float32),
        'stride': [1, 2, 1],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [1, 1, 1],
        'name': 'conv1d_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'input': np.random.uniform(-5.0, -1.0, (2, 8, 2)).astype(np.float32),
        'filters': np.random.uniform(-2.0, 2.0, (3, 2, 3)).astype(np.float32),
        'stride': [1],
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'conv1d_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'input': np.random.randn(1, 100, 16).astype(np.float32),
        'filters': np.random.randn(7, 16, 32).astype(np.float32),
        'stride': [2],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'conv1d_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'input': np.random.randn(2, 10, 3).astype(np.float32),
        'filters': np.random.randn(3, 3, 4).astype(np.float32),
        'stride': [1],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [1, 2, 1],
        'name': 'conv1d_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.conv1d_2"] = tf_nn_conv1d_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_conv1d_inputs():
    list_of_inputs = []
    
    # Input 1: NWC, float32, stride 1, padding SAME, dilation 1
    input_dict = {
        'input': np.random.randn(2, 10, 3).astype(np.float32),
        'filters': np.random.randn(3, 3, 5).astype(np.float32),
        'stride': 1,
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: NWC, float32, stride 2, padding VALID, dilation 1
    input_dict = {
        'input': np.random.randn(1, 15, 4).astype(np.float32),
        'filters': np.random.randn(2, 4, 2).astype(np.float32),
        'stride': 2,
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: NWC, float64, stride 1, padding SAME, dilation 2
    input_dict = {
        'input': np.random.randn(4, 8, 2).astype(np.float64),
        'filters': np.random.randn(3, 2, 4).astype(np.float64),
        'stride': 1,
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 2,
        'name': 'conv1d_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: NWC, float32, stride 2, padding VALID, dilation 2
    input_dict = {
        'input': np.random.randn(2, 12, 3).astype(np.float32),
        'filters': np.random.randn(3, 3, 3).astype(np.float32),
        'stride': 2,
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': 2,
        'name': 'conv1d_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: NWC, float32, stride 3, padding SAME, dilation 1
    input_dict = {
        'input': np.random.randn(3, 20, 1).astype(np.float32),
        'filters': np.random.randn(5, 1, 8).astype(np.float32),
        'stride': 3,
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: NWC, float32, stride 1, padding VALID, dilation 3
    input_dict = {
        'input': np.random.randn(2, 5, 4).astype(np.float32),
        'filters': np.random.randn(1, 4, 4).astype(np.float32),
        'stride': 1,
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': 3,
        'name': 'conv1d_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: NWC, float64, stride 2, padding SAME, dilation 1
    input_dict = {
        'input': np.random.randn(1, 30, 2).astype(np.float64),
        'filters': np.random.randn(4, 2, 2).astype(np.float64),
        'stride': 2,
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: NWC, float32, stride 1, padding VALID, dilation 1
    input_dict = {
        'input': np.random.randn(5, 7, 8).astype(np.float32),
        'filters': np.random.randn(2, 8, 16).astype(np.float32),
        'stride': 1,
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 4D batch_shape NWC, float32, stride 1, padding SAME, dilation 1
    input_dict = {
        'input': np.random.randn(2, 2, 10, 3).astype(np.float32),
        'filters': np.random.randn(3, 3, 5).astype(np.float32),
        'stride': 1,
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Explicit values, float32, NWC, stride 1, padding VALID, dilation 1
    input_dict = {
        'input': np.array([[[-1.0, 2.0], [0.5, -1.5], [2.0, -3.0]]], dtype=np.float32),
        'filters': np.array([[[1.0, -1.0], [2.0, 0.0]]], dtype=np.float32),
        'stride': 1,
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.conv1d_3"] = generate_conv1d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_conv1d_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 NWC valid combination with 'SAME' padding
    input_val = np.random.randn(2, 10, 3).astype(np.float32)
    filters_val = np.random.randn(3, 3, 4).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filters': filters_val,
        'stride': [1],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: NWC format with float32, stride [2], and 'VALID' padding
    input_val = np.random.randn(2, 10, 3).astype(np.float32)
    filters_val = np.random.randn(3, 3, 4).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filters': filters_val,
        'stride': [2],
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: NWC format, float16 with dilation 2
    input_val = np.random.randn(1, 8, 2).astype(np.float16)
    filters_val = np.random.randn(2, 2, 2).astype(np.float16)
    input_dict = {
        'input': input_val,
        'filters': filters_val,
        'stride': [1],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 2,
        'name': 'conv1d_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: NWC format, float64, standard parameters
    input_val = np.random.randn(4, 16, 4).astype(np.float64)
    filters_val = np.random.randn(3, 4, 8).astype(np.float64)
    input_dict = {
        'input': input_val,
        'filters': filters_val,
        'stride': [1],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: NWC format with negative values in inputs
    input_val = np.array([[[-1.0, -2.0], [3.0, 4.0], [-5.0, -6.0]]], dtype=np.float32)
    filters_val = np.array([[[1.0, -1.0], [-1.0, 1.0]], [[2.0, -2.0], [-2.0, 2.0]]], dtype=np.float32)
    input_dict = {
        'input': input_val,
        'filters': filters_val,
        'stride': [1],
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Higher dimension batch (batch_shape = [2, 3]), NWC format
    input_val = np.random.randn(2, 3, 5, 2).astype(np.float32)
    filters_val = np.random.randn(2, 2, 4).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filters': filters_val,
        'stride': [1],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Higher dimension batch (batch_shape = [2, 2]), NWC format
    input_val = np.random.randn(2, 2, 8, 3).astype(np.float32)
    filters_val = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filters': filters_val,
        'stride': [2],
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger stride [3]
    input_val = np.random.randn(1, 15, 3).astype(np.float32)
    filters_val = np.random.randn(3, 3, 3).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filters': filters_val,
        'stride': [3],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Dilation of 3
    input_val = np.random.randn(1, 12, 2).astype(np.float32)
    filters_val = np.random.randn(3, 2, 2).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filters': filters_val,
        'stride': [1],
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': 3,
        'name': 'conv1d_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger channel sizes
    input_val = np.random.randn(1, 5, 64).astype(np.float32)
    filters_val = np.random.randn(3, 64, 32).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filters': filters_val,
        'stride': [1],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': 1,
        'name': 'conv1d_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.conv1d_4"] = tf_nn_conv1d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_conv2d_inputs():
    list_of_inputs = []

    # Input 1: Standard NHWC, float32, SAME padding
    input_dict = {
        'input': np.random.randn(1, 16, 16, 3).astype(np.float32),
        'filters': np.random.randn(3, 3, 3, 16).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: NHWC, float16, VALID padding, strides of 2
    input_dict = {
        'input': np.random.randn(2, 10, 10, 1).astype(np.float16),
        'filters': np.random.randn(2, 2, 1, 8).astype(np.float16),
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: NHWC format, float64, strides length 4
    input_dict = {
        'input': np.random.randn(1, 32, 32, 4).astype(np.float64),
        'filters': np.random.randn(3, 3, 4, 8).astype(np.float64),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: NHWC, dilated convolution with dilation 2
    input_dict = {
        'input': np.random.randn(4, 16, 16, 3).astype(np.float32),
        'filters': np.random.randn(3, 3, 3, 8).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 2, 2, 1],
        'name': 'conv_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values, NHWC, float32, VALID padding
    input_dict = {
        'input': (np.ones((1, 8, 8, 1), dtype=np.float32) * -1.5),
        'filters': (np.ones((2, 2, 1, 1), dtype=np.float32) * 0.5),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Higher dimension batch (5-D input)
    input_dict = {
        'input': np.random.randn(2, 3, 10, 10, 3).astype(np.float32),
        'filters': np.random.randn(2, 2, 3, 4).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Strides of length 2
    input_dict = {
        'input': np.random.randn(1, 14, 14, 4).astype(np.float32),
        'filters': np.random.randn(3, 3, 4, 16).astype(np.float32),
        'strides': [2, 2],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'conv_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Strides of length 1 (replicated to H and W)
    input_dict = {
        'input': np.random.randn(2, 8, 8, 2).astype(np.float32),
        'filters': np.random.randn(2, 2, 2, 4).astype(np.float32),
        'strides': [1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1],
        'name': 'conv_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: High dimension batch (6-D input)
    input_dict = {
        'input': np.random.randn(1, 2, 2, 12, 12, 3).astype(np.float32),
        'filters': np.random.randn(3, 3, 3, 6).astype(np.float32),
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1x1 Convolution with many channels
    input_dict = {
        'input': np.random.randn(1, 8, 8, 64).astype(np.float32),
        'filters': np.random.randn(1, 1, 64, 128).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.conv2d"] = tf_nn_conv2d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_conv3d_inputs():
    list_of_inputs = []

    # Case 1: Simple valid case with NDHWC, same padding, float32
    input_val = np.random.randn(1, 2, 2, 2, 1).astype(np.float32)
    filters = np.random.randn(1, 1, 1, 1, 1).astype(np.float32)
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"
    data_format = "NDHWC"
    dilations = [1, 1, 1, 1, 1]
    name = "conv1"
    list_of_inputs.append({
        'input': input_val, 'filters': filters, 'strides': strides,
        'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name
    })

    # Case 2: Larger shapes, negative values, valid padding, float32
    input_val = np.random.randn(2, 3, 3, 3, 2).astype(np.float32)
    filters = np.random.randn(2, 2, 2, 2, 4).astype(np.float32)
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    dilations = [1, 1, 1, 1, 1]
    name = "conv2"
    list_of_inputs.append({
        'input': input_val, 'filters': filters, 'strides': strides,
        'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name
    })

    # Case 3: Strides of 2 in spatial dimensions, same padding, float32
    input_val = np.random.randn(1, 4, 4, 4, 3).astype(np.float32)
    filters = np.random.randn(3, 3, 3, 3, 2).astype(np.float32)
    strides = [1, 2, 2, 2, 1]
    padding = "SAME"
    data_format = "NDHWC"
    dilations = [1, 1, 1, 1, 1]
    name = "conv3"
    list_of_inputs.append({
        'input': input_val, 'filters': filters, 'strides': strides,
        'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name
    })

    # Case 4: Multiple input/output channels, NDHWC, float32
    input_val = np.random.randn(1, 3, 3, 3, 2).astype(np.float32)
    filters = np.random.randn(2, 2, 2, 2, 3).astype(np.float32)
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"
    data_format = "NDHWC"
    dilations = [1, 1, 1, 1, 1]
    name = "conv4"
    list_of_inputs.append({
        'input': input_val, 'filters': filters, 'strides': strides,
        'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name
    })

    # Case 5: Float16 precision
    input_val = np.random.randn(1, 2, 2, 2, 1).astype(np.float16)
    filters = np.random.randn(1, 1, 1, 1, 1).astype(np.float16)
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    dilations = [1, 1, 1, 1, 1]
    name = "conv5"
    list_of_inputs.append({
        'input': input_val, 'filters': filters, 'strides': strides,
        'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name
    })

    # Case 6: Float64 precision
    input_val = np.random.randn(1, 3, 3, 3, 1).astype(np.float64)
    filters = np.random.randn(2, 2, 2, 1, 2).astype(np.float64)
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"
    data_format = "NDHWC"
    dilations = [1, 1, 1, 1, 1]
    name = "conv6"
    list_of_inputs.append({
        'input': input_val, 'filters': filters, 'strides': strides,
        'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name
    })

    # Case 7: NDHWC
    input_val = np.random.randn(1, 5, 5, 5, 1).astype(np.float32)
    filters = np.random.randn(2, 2, 2, 1, 1).astype(np.float32)
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"
    data_format = "NDHWC"
    dilations = [1, 1, 1, 1, 1]
    name = "conv7"
    list_of_inputs.append({
        'input': input_val, 'filters': filters, 'strides': strides,
        'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name
    })

    # Case 8: Heterogeneous strides (different stride for height/width)
    input_val = np.random.randn(2, 4, 4, 4, 3).astype(np.float32)
    filters = np.random.randn(1, 2, 3, 3, 2).astype(np.float32)
    strides = [1, 1, 2, 2, 1]
    padding = "VALID"
    data_format = "NDHWC"
    dilations = [1, 1, 1, 1, 1]
    name = "conv8"
    list_of_inputs.append({
        'input': input_val, 'filters': filters, 'strides': strides,
        'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name
    })

    # Case 9: Stride on depth dimension only
    input_val = np.random.randn(1, 6, 6, 6, 2).astype(np.float32)
    filters = np.random.randn(3, 3, 3, 2, 2).astype(np.float32)
    strides = [1, 2, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    dilations = [1, 1, 1, 1, 1]
    name = "conv9"
    list_of_inputs.append({
        'input': input_val, 'filters': filters, 'strides': strides,
        'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name
    })

    # Case 10: Mixed positive and negative float32 values with simple strides
    input_val = np.random.uniform(-1, 1, (1, 3, 3, 3, 2)).astype(np.float32)
    filters = np.random.uniform(-1, 1, (2, 2, 2, 2, 1)).astype(np.float32)
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    dilations = [1, 1, 1, 1, 1]
    name = "conv10"
    list_of_inputs.append({
        'input': input_val, 'filters': filters, 'strides': strides,
        'padding': padding, 'data_format': data_format, 'dilations': dilations, 'name': name
    })

    return list_of_inputs

generated_inputs["tf.nn.conv3d"] = tf_nn_conv3d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_ctc_beam_search_decoder_inputs():
    list_of_inputs = []

    # Input 1: Standard small configuration
    inputs = np.random.randn(5, 2, 3).astype(np.float32)
    seq_len = np.array([5, 4], dtype=np.int32)
    list_of_inputs.append({
        'inputs': inputs,
        'sequence_length': seq_len,
        'beam_width': 10,
        'top_paths': 2
    })

    # Input 2: Minimal configuration
    inputs = np.random.randn(1, 1, 2).astype(np.float32)
    seq_len = np.array([1], dtype=np.int32)
    list_of_inputs.append({
        'inputs': inputs,
        'sequence_length': seq_len,
        'beam_width': 1,
        'top_paths': 1
    })

    # Input 3: Medium configuration with beam_width equal to top_paths
    inputs = np.random.randn(10, 4, 8).astype(np.float32)
    seq_len = np.array([8, 10, 5, 9], dtype=np.int32)
    list_of_inputs.append({
        'inputs': inputs,
        'sequence_length': seq_len,
        'beam_width': 5,
        'top_paths': 5
    })

    # Input 4: Large number of classes with float64 inputs
    inputs = np.random.randn(12, 3, 50).astype(np.float64)
    seq_len = np.array([12, 10, 11], dtype=np.int32)
    list_of_inputs.append({
        'inputs': inputs,
        'sequence_length': seq_len,
        'beam_width': 20,
        'top_paths': 3
    })

    # Input 5: High beam width and multiple top paths
    inputs = np.random.randn(8, 2, 10).astype(np.float32)
    seq_len = np.array([6, 8], dtype=np.int32)
    list_of_inputs.append({
        'inputs': inputs,
        'sequence_length': seq_len,
        'beam_width': 100,
        'top_paths': 10
    })

    # Input 6: Variable small sequence lengths
    inputs = np.random.randn(15, 5, 6).astype(np.float32)
    seq_len = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    list_of_inputs.append({
        'inputs': inputs,
        'sequence_length': seq_len,
        'beam_width': 8,
        'top_paths': 4
    })

    # Input 7: Larger batch size with uniform sequence length
    inputs = np.random.randn(6, 16, 4).astype(np.float32)
    seq_len = np.array([6] * 16, dtype=np.int32)
    list_of_inputs.append({
        'inputs': inputs,
        'sequence_length': seq_len,
        'beam_width': 4,
        'top_paths': 2
    })

    # Input 8: Hand-crafted deterministic values
    inputs = np.ones((4, 2, 5), dtype=np.float32) * -1.5
    inputs[0, 0, 1] = 2.0
    inputs[1, 1, 2] = 3.5
    seq_len = np.array([4, 3], dtype=np.int32)
    list_of_inputs.append({
        'inputs': inputs,
        'sequence_length': seq_len,
        'beam_width': 3,
        'top_paths': 1
    })

    # Input 9: Uniformly distributed inputs
    inputs = np.random.uniform(-10.0, 10.0, (7, 3, 12)).astype(np.float32)
    seq_len = np.array([7, 6, 5], dtype=np.int32)
    list_of_inputs.append({
        'inputs': inputs,
        'sequence_length': seq_len,
        'beam_width': 12,
        'top_paths': 6
    })

    # Input 10: Single top path with substantial beam width
    inputs = np.random.randn(20, 2, 15).astype(np.float32)
    seq_len = np.array([20, 18], dtype=np.int32)
    list_of_inputs.append({
        'inputs': inputs,
        'sequence_length': seq_len,
        'beam_width': 50,
        'top_paths': 1
    })

    return list_of_inputs

generated_inputs["tf.nn.ctc_beam_search_decoder"] = tf_nn_ctc_beam_search_decoder_inputs()

import copy
import numpy as np
import tensorflow as tf


def tf_nn_ctc_greedy_decoder_inputs():
    list_of_inputs = []

    # Input 1
    inputs = np.random.randn(5, 2, 3).astype(np.float32)
    sequence_length = np.array([5, 4], dtype=np.int32)
    merge_repeated = True
    blank_index = 2

    input_dict = {
        "inputs": inputs,
        "sequence_length": sequence_length,
        "merge_repeated": merge_repeated,
        "blank_index": blank_index,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = np.random.randn(10, 3, 5).astype(np.float32)
    sequence_length = np.array([10, 8, 9], dtype=np.int32)
    merge_repeated = False
    blank_index = 0

    input_dict = {
        "inputs": inputs,
        "sequence_length": sequence_length,
        "merge_repeated": merge_repeated,
        "blank_index": blank_index,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = np.random.randn(2, 1, 2).astype(np.float32)
    sequence_length = np.array([2], dtype=np.int32)
    merge_repeated = True
    blank_index = -1

    input_dict = {
        "inputs": inputs,
        "sequence_length": sequence_length,
        "merge_repeated": merge_repeated,
        "blank_index": blank_index,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inputs = np.random.randn(15, 4, 10).astype(np.float32)
    sequence_length = np.array([12, 15, 10, 14], dtype=np.int32)
    merge_repeated = False
    blank_index = 9

    input_dict = {
        "inputs": inputs,
        "sequence_length": sequence_length,
        "merge_repeated": merge_repeated,
        "blank_index": blank_index,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    inputs = np.random.randn(8, 2, 4).astype(np.float32)
    sequence_length = np.array([8, 8], dtype=np.int32)
    merge_repeated = True
    blank_index = 3

    input_dict = {
        "inputs": inputs,
        "sequence_length": sequence_length,
        "merge_repeated": merge_repeated,
        "blank_index": blank_index,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    inputs = np.random.randn(6, 2, 3).astype(np.float32)
    sequence_length = np.array([6, 5], dtype=np.int32)
    merge_repeated = True
    blank_index = 1

    input_dict = {
        "inputs": inputs,
        "sequence_length": sequence_length,
        "merge_repeated": merge_repeated,
        "blank_index": blank_index,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    inputs = np.random.randn(4, 5, 3).astype(np.float32)
    sequence_length = np.array([4, 3, 2, 4, 1], dtype=np.int32)
    merge_repeated = False
    blank_index = 2

    input_dict = {
        "inputs": inputs,
        "sequence_length": sequence_length,
        "merge_repeated": merge_repeated,
        "blank_index": blank_index,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    inputs = np.random.randn(3, 3, 3).astype(np.float32)
    sequence_length = np.array([3, 2, 1], dtype=np.int32)
    merge_repeated = True
    blank_index = -1

    input_dict = {
        "inputs": inputs,
        "sequence_length": sequence_length,
        "merge_repeated": merge_repeated,
        "blank_index": blank_index,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    inputs = np.random.randn(20, 1, 50).astype(np.float32)
    sequence_length = np.array([20], dtype=np.int32)
    merge_repeated = False
    blank_index = 49

    input_dict = {
        "inputs": inputs,
        "sequence_length": sequence_length,
        "merge_repeated": merge_repeated,
        "blank_index": blank_index,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    inputs = np.random.randn(7, 3, 6).astype(np.float32)
    sequence_length = np.array([5, 6, 7], dtype=np.int32)
    merge_repeated = True
    blank_index = 4

    input_dict = {
        "inputs": inputs,
        "sequence_length": sequence_length,
        "merge_repeated": merge_repeated,
        "blank_index": blank_index,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["tf.nn.ctc_greedy_decoder"] = tf_nn_ctc_greedy_decoder_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_depthwise_conv2d_inputs():
    list_of_inputs = []

    # Input 1: Standard case NHWC, stride 1, no dilation, valid padding, multiplier 1
    list_of_inputs.append({
        'input': np.random.randn(2, 5, 5, 3).astype(np.float32),
        'filter': np.random.randn(3, 3, 3, 1).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'depthwise_conv_1'
    })

    # Input 2: NHWC, stride 2, valid padding, multiplier 2
    list_of_inputs.append({
        'input': np.random.randn(1, 10, 10, 2).astype(np.float32),
        'filter': np.random.randn(3, 3, 2, 2).astype(np.float32),
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'depthwise_conv_2'
    })

    # Input 3: NHWC, SAME padding, stride 1, multiplier 1
    list_of_inputs.append({
        'input': np.random.randn(4, 8, 8, 4).astype(np.float32),
        'filter': np.random.randn(5, 5, 4, 1).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'depthwise_conv_3'
    })

    # Input 4: NHWC format, SAME padding, stride 1
    list_of_inputs.append({
        'input': np.random.randn(2, 16, 16, 3).astype(np.float32),
        'filter': np.random.randn(3, 3, 3, 2).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'depthwise_conv_4'
    })

    # Input 5: Dilated NHWC, stride 1, SAME padding, dilation [2, 2]
    list_of_inputs.append({
        'input': np.random.randn(1, 14, 14, 3).astype(np.float32),
        'filter': np.random.randn(3, 3, 3, 1).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [2, 2],
        'name': 'depthwise_conv_5'
    })

    # Input 6: Negative values, NHWC, VALID, stride 1, multiplier 3
    list_of_inputs.append({
        'input': -np.random.rand(2, 4, 4, 2).astype(np.float32),
        'filter': -np.random.rand(2, 2, 2, 3).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'depthwise_conv_6'
    })

    # Input 7: Float64 type
    list_of_inputs.append({
        'input': np.random.randn(1, 5, 5, 1).astype(np.float64),
        'filter': np.random.randn(3, 3, 1, 2).astype(np.float64),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'depthwise_conv_7'
    })

    # Input 8: High dilation, NHWC
    list_of_inputs.append({
        'input': np.random.randn(1, 32, 32, 2).astype(np.float32),
        'filter': np.random.randn(5, 5, 2, 1).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [3, 3],
        'name': 'depthwise_conv_8'
    })

    # Input 9: Stride 3, SAME padding
    list_of_inputs.append({
        'input': np.random.randn(2, 15, 15, 3).astype(np.float32),
        'filter': np.random.randn(3, 3, 3, 2).astype(np.float32),
        'strides': [1, 3, 3, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'depthwise_conv_9'
    })

    # Input 10: Multiplier 5, small height/width
    list_of_inputs.append({
        'input': np.random.randn(3, 3, 3, 4).astype(np.float32),
        'filter': np.random.randn(1, 1, 4, 5).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'depthwise_conv_10'
    })

    return list_of_inputs

generated_inputs["tf.nn.depthwise_conv2d"] = tf_nn_depthwise_conv2d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_dilation2d_inputs():
    list_of_inputs = []

    # Input 1: Float32, SAME padding, stride 1, dilation 1
    input_dict = {
        'input': np.random.randn(1, 5, 5, 1).astype(np.float32),
        'filters': np.random.randn(3, 3, 1).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': "SAME",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "dilation_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float64, VALID padding, stride 2, dilation 1
    input_dict = {
        'input': np.random.randn(2, 6, 6, 3).astype(np.float64),
        'filters': np.random.randn(2, 2, 3).astype(np.float64),
        'strides': [1, 2, 2, 1],
        'padding': "VALID",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "dilation_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Int32, SAME padding, stride 1, dilation 2
    input_dict = {
        'input': np.random.randint(-10, 10, size=(1, 5, 5, 2)).astype(np.int32),
        'filters': np.random.randint(-5, 5, size=(2, 2, 2)).astype(np.int32),
        'strides': [1, 1, 1, 1],
        'padding': "SAME",
        'data_format': "NHWC",
        'dilations': [1, 2, 2, 1],
        'name': "dilation_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Uint8, VALID padding, stride 1, dilation 1 (positive values only)
    input_dict = {
        'input': np.random.randint(0, 255, size=(1, 4, 4, 1)).astype(np.uint8),
        'filters': np.random.randint(0, 10, size=(2, 2, 1)).astype(np.uint8),
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "dilation_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float32, negative values, SAME padding, stride 1
    input_dict = {
        'input': (np.ones((1, 3, 3, 1), dtype=np.float32) * -5.0),
        'filters': np.zeros((2, 2, 1), dtype=np.float32),
        'strides': [1, 1, 1, 1],
        'padding': "SAME",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "dilation_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float32, VALID padding, stride 1, dilation 1
    input_dict = {
        'input': np.random.randn(2, 8, 8, 4).astype(np.float32),
        'filters': np.random.randn(3, 3, 4).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "dilation_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Int32, SAME padding, stride 2, dilation 1
    input_dict = {
        'input': np.random.randint(-50, 50, size=(1, 10, 10, 2)).astype(np.int32),
        'filters': np.random.randint(-5, 5, size=(3, 3, 2)).astype(np.int32),
        'strides': [1, 2, 2, 1],
        'padding': "SAME",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "dilation_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Int32, VALID padding, stride 1, dilation 2
    input_dict = {
        'input': np.random.randint(0, 1000, size=(1, 7, 7, 3)).astype(np.int32),
        'filters': np.random.randint(0, 50, size=(2, 2, 3)).astype(np.int32),
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'data_format': "NHWC",
        'dilations': [1, 2, 2, 1],
        'name': "dilation_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Float32, larger spatial dimensions, SAME padding, stride 3
    input_dict = {
        'input': np.random.randn(1, 15, 15, 1).astype(np.float32),
        'filters': np.random.randn(3, 3, 1).astype(np.float32),
        'strides': [1, 3, 3, 1],
        'padding': "SAME",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "dilation_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float32, asymmetric strides, VALID padding
    input_dict = {
        'input': np.random.randn(1, 6, 8, 2).astype(np.float32),
        'filters': np.random.randn(2, 3, 2).astype(np.float32),
        'strides': [1, 2, 1, 1],
        'padding': "VALID",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "dilation_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.dilation2d"] = tf_nn_dilation2d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_erosion2d_inputs():
    list_of_inputs = []

    # 1. Standard float32 VALID
    list_of_inputs.append({
        'value': np.random.uniform(-10, 10, (1, 3, 3, 1)).astype(np.float32),
        'filters': np.random.uniform(-5, 5, (2, 2, 1)).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "erosion_1"
    })

    # 2. Standard float32 SAME with strides
    list_of_inputs.append({
        'value': np.random.uniform(-10, 10, (2, 5, 5, 3)).astype(np.float32),
        'filters': np.random.uniform(-5, 5, (3, 3, 3)).astype(np.float32),
        'strides': [1, 2, 2, 1],
        'padding': "SAME",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "erosion_2"
    })

    # 3. float64 with VALID and dilations
    list_of_inputs.append({
        'value': np.random.uniform(-10, 10, (1, 10, 10, 2)).astype(np.float64),
        'filters': np.random.uniform(-5, 5, (3, 3, 2)).astype(np.float64),
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'data_format': "NHWC",
        'dilations': [1, 2, 2, 1],
        'name': "erosion_3"
    })

    # 4. Negatives, SAME padding, stride and dilations
    list_of_inputs.append({
        'value': np.random.uniform(-50, -10, (2, 8, 8, 4)).astype(np.float32),
        'filters': np.random.uniform(-5, 0, (2, 2, 4)).astype(np.float32),
        'strides': [1, 2, 1, 1],
        'padding': "SAME",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "erosion_4"
    })

    # 5. float32 with VALID, dilations, and strides=1
    list_of_inputs.append({
        'value': np.random.uniform(0, 255, (1, 6, 6, 1)).astype(np.float32),
        'filters': np.random.uniform(0, 10, (3, 3, 1)).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'data_format': "NHWC",
        'dilations': [1, 2, 2, 1],
        'name': "erosion_5"
    })

    # 6. float64, SAME padding
    list_of_inputs.append({
        'value': np.random.uniform(-1, 1, (1, 4, 4, 1)).astype(np.float64),
        'filters': np.random.uniform(-1, 1, (2, 2, 1)).astype(np.float64),
        'strides': [1, 1, 1, 1],
        'padding': "SAME",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "erosion_6"
    })

    # 7. Large strides, float32
    list_of_inputs.append({
        'value': np.random.uniform(-5, 5, (4, 7, 7, 3)).astype(np.float32),
        'filters': np.random.uniform(-2, 2, (3, 3, 3)).astype(np.float32),
        'strides': [1, 3, 3, 1],
        'padding': "SAME",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "erosion_7"
    })

    # 8. 1x1 filter, float32, VALID
    list_of_inputs.append({
        'value': np.random.uniform(-10, 10, (1, 5, 5, 1)).astype(np.float32),
        'filters': np.random.uniform(-5, 5, (1, 1, 1)).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "erosion_8"
    })

    # 9. Large input with dilations and strides
    list_of_inputs.append({
        'value': np.random.uniform(-100, 100, (2, 12, 12, 2)).astype(np.float32),
        'filters': np.random.uniform(-10, 10, (4, 4, 2)).astype(np.float32),
        'strides': [1, 2, 2, 1],
        'padding': "VALID",
        'data_format': "NHWC",
        'dilations': [1, 2, 2, 1],
        'name': "erosion_9"
    })

    # 10. Non-square input and filter shapes, float64
    list_of_inputs.append({
        'value': np.random.uniform(-5, 5, (1, 3, 5, 2)).astype(np.float64),
        'filters': np.random.uniform(-1, 1, (2, 3, 2)).astype(np.float64),
        'strides': [1, 1, 1, 1],
        'padding': "SAME",
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "erosion_10"
    })

    return list_of_inputs

generated_inputs["tf.nn.erosion2d"] = tf_nn_erosion2d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_fractional_avg_pool_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "value": np.random.rand(1, 10, 10, 3).astype(np.float32),
        "pooling_ratio": [1.0, 1.5, 1.5, 1.0],
        "pseudo_random": False,
        "overlapping": False,
        "seed": 42,
        "name": "avg_pool_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "value": np.random.rand(2, 20, 20, 1).astype(np.float64),
        "pooling_ratio": [1.0, 2.0, 2.0, 1.0],
        "pseudo_random": True,
        "overlapping": True,
        "seed": 10,
        "name": "avg_pool_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "value": np.ones((1, 8, 8, 16), dtype=np.float32),
        "pooling_ratio": [1.0, 1.2, 1.3, 1.0],
        "pseudo_random": False,
        "overlapping": True,
        "seed": 0,
        "name": "avg_pool_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "value": np.random.randn(4, 15, 15, 3).astype(np.float32),
        "pooling_ratio": [1.0, 1.1, 1.1, 1.0],
        "pseudo_random": True,
        "overlapping": False,
        "seed": 123,
        "name": "avg_pool_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "value": np.zeros((1, 100, 100, 3), dtype=np.float32),
        "pooling_ratio": [1.0, 1.0, 1.0, 1.0],
        "pseudo_random": False,
        "overlapping": False,
        "seed": 99,
        "name": "avg_pool_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "value": np.random.rand(1, 5, 5, 2).astype(np.float32),
        "pooling_ratio": [1.0, 1.8, 1.8, 1.0],
        "pseudo_random": True,
        "overlapping": True,
        "seed": 5,
        "name": "avg_pool_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "value": (np.random.rand(2, 12, 12, 4) * 10).astype(np.float32),
        "pooling_ratio": [1.0, 1.44, 1.73, 1.0],
        "pseudo_random": False,
        "overlapping": False,
        "seed": 7,
        "name": "avg_pool_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "value": np.random.randn(3, 30, 30, 8).astype(np.float32),
        "pooling_ratio": [1.0, 2.5, 2.5, 1.0],
        "pseudo_random": True,
        "overlapping": False,
        "seed": 1001,
        "name": "avg_pool_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "value": np.arange(144, dtype=np.float32).reshape((1, 12, 12, 1)),
        "pooling_ratio": [1.0, 1.5, 1.2, 1.0],
        "pseudo_random": False,
        "overlapping": True,
        "seed": 88,
        "name": "avg_pool_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "value": np.random.uniform(-1, 1, (2, 16, 16, 3)).astype(np.float32),
        "pooling_ratio": [1.0, 1.33, 1.33, 1.0],
        "pseudo_random": True,
        "overlapping": True,
        "seed": 55,
        "name": "avg_pool_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.fractional_avg_pool"] = tf_nn_fractional_avg_pool_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_fractional_max_pool_inputs():
    list_of_inputs = []
    
    # Input 1
    input_dict = {
        'value': np.random.randn(1, 10, 10, 1).astype(np.float32),
        'pooling_ratio': [1.0, 1.5, 1.5, 1.0],
        'pseudo_random': True,
        'overlapping': True,
        'seed': 42,
        'name': "pool1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input_dict = {
        'value': np.random.randn(2, 20, 20, 3).astype(np.float32),
        'pooling_ratio': [1.0, 1.2, 1.8, 1.0],
        'pseudo_random': False,
        'overlapping': False,
        'seed': 1,
        'name': "pool2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input_dict = {
        'value': np.random.randint(-10, 10, size=(1, 5, 5, 1)).astype(np.int32),
        'pooling_ratio': [1.0, 1.1, 1.1, 1.0],
        'pseudo_random': True,
        'overlapping': False,
        'seed': 10,
        'name': "pool3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_dict = {
        'value': np.random.randn(4, 15, 15, 2).astype(np.float64),
        'pooling_ratio': [1.0, 1.44, 1.73, 1.0],
        'pseudo_random': False,
        'overlapping': True,
        'seed': 100,
        'name': "pool4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_dict = {
        'value': np.random.randn(1, 100, 100, 3).astype(np.float32),
        'pooling_ratio': [1.0, 5.5, 5.5, 1.0],
        'pseudo_random': True,
        'overlapping': True,
        'seed': 7,
        'name': "pool5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_dict = {
        'value': np.random.randint(-50, 50, size=(3, 8, 8, 4)).astype(np.int64),
        'pooling_ratio': [1.0, 1.0, 1.0, 1.0],
        'pseudo_random': False,
        'overlapping': False,
        'seed': 88,
        'name': "pool6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_dict = {
        'value': np.random.randn(1, 50, 50, 1).astype(np.float32),
        'pooling_ratio': [1.0, 2.5, 3.5, 1.0],
        'pseudo_random': True,
        'overlapping': False,
        'seed': 999,
        'name': "pool7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_dict = {
        'value': np.random.randn(2, 12, 16, 3).astype(np.float32),
        'pooling_ratio': [1.0, 1.3, 1.4, 1.0],
        'pseudo_random': False,
        'overlapping': True,
        'seed': 12345,
        'name': "pool8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_dict = {
        'value': np.random.randn(1, 7, 9, 2).astype(np.float32),
        'pooling_ratio': [1.0, 1.05, 1.15, 1.0],
        'pseudo_random': True,
        'overlapping': True,
        'seed': 3,
        'name': "pool9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_dict = {
        'value': np.random.randn(5, 25, 25, 5).astype(np.float32),
        'pooling_ratio': [1.0, 2.1, 2.1, 1.0],
        'pseudo_random': False,
        'overlapping': False,
        'seed': 12,
        'name': "pool10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.fractional_max_pool"] = tf_nn_fractional_max_pool_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool_inputs():
    list_of_inputs = []
    
    # Input 1: 2D Spatial (NHWC), float32, SAME padding
    list_of_inputs.append({
        'input': np.random.randn(2, 4, 4, 3).astype(np.float32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': "SAME",
        'data_format': "NHWC",
        'name': "pool1"
    })
    
    # Input 2: 2D Spatial (NHWC), float32, VALID padding
    list_of_inputs.append({
        'input': np.random.randn(2, 5, 5, 3).astype(np.float32),
        'ksize': [1, 3, 3, 1],
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'data_format': "NHWC",
        'name': "pool2"
    })

    # Input 3: 1D Spatial (NWC), float32, SAME padding
    list_of_inputs.append({
        'input': np.random.randn(2, 8, 3).astype(np.float32),
        'ksize': [1, 2, 1],
        'strides': [1, 2, 1],
        'padding': "SAME",
        'data_format': "NWC",
        'name': "pool3"
    })

    # Input 4: 1D Spatial (NWC), float32, VALID padding
    list_of_inputs.append({
        'input': np.random.randn(2, 8, 3).astype(np.float32),
        'ksize': [1, 3, 1],
        'strides': [1, 1, 1],
        'padding': "VALID",
        'data_format': "NWC",
        'name': "pool4"
    })

    # Input 5: 3D Spatial (NDHWC), float32, SAME padding
    list_of_inputs.append({
        'input': np.random.randn(2, 4, 4, 4, 3).astype(np.float32),
        'ksize': [1, 2, 2, 2, 1],
        'strides': [1, 2, 2, 2, 1],
        'padding': "SAME",
        'data_format': "NDHWC",
        'name': "pool5"
    })

    # Input 6: 3D Spatial (NDHWC), float32, VALID padding
    list_of_inputs.append({
        'input': np.random.randn(2, 4, 4, 4, 3).astype(np.float32),
        'ksize': [1, 3, 3, 3, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': "VALID",
        'data_format': "NDHWC",
        'name': "pool6"
    })

    # Input 7: 2D Spatial (NHWC) with negative values
    list_of_inputs.append({
        'input': np.array([[[[-1.0, 2.0], [3.0, -4.0]], [[5.0, -6.0], [-7.0, 8.0]]]], dtype=np.float32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'data_format': "NHWC",
        'name': "pool7"
    })

    # Input 8: 2D Spatial (NHWC) with length 1 ksize and strides
    list_of_inputs.append({
        'input': np.random.randn(2, 4, 4, 3).astype(np.float32),
        'ksize': [2],
        'strides': [2],
        'padding': "SAME",
        'data_format': "NHWC",
        'name': "pool8"
    })

    # Input 9: 2D Spatial (NHWC) with length N ksize and strides
    list_of_inputs.append({
        'input': np.random.randn(2, 6, 6, 3).astype(np.float32),
        'ksize': [3, 3],
        'strides': [1, 1],
        'padding': "SAME",
        'data_format': "NHWC",
        'name': "pool9"
    })

    # Input 10: 1D Spatial (NWC) with length 1 ksize and strides
    list_of_inputs.append({
        'input': np.random.randn(1, 10, 2).astype(np.float32),
        'ksize': [3],
        'strides': [2],
        'padding': "VALID",
        'data_format': "NWC",
        'name': "pool10"
    })

    return list_of_inputs

generated_inputs["tf.nn.max_pool"] = tf_nn_max_pool_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool1d_inputs():
    list_of_inputs = []

    # Case 1: Standard NWC, float32, VALID padding, ksize=2, stride=1
    input_dict = {
        'input': np.random.randn(2, 10, 3).astype(np.float32),
        'ksize': [2],
        'strides': [1],
        'padding': "VALID",
        'data_format': "NWC",
        'name': "pool_case_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: NWC, float32, SAME padding, ksize=3, stride=2, negative values
    input_dict = {
        'input': np.random.uniform(-5.0, -1.0, (1, 8, 2)).astype(np.float32),
        'ksize': [3],
        'strides': [2],
        'padding': "SAME",
        'data_format': "NWC",
        'name': "pool_case_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: NWC, float32, VALID padding, ksize=[1, 2, 1]
    input_dict = {
        'input': np.random.randn(4, 8, 2).astype(np.float32),
        'ksize': [1, 2, 1],
        'strides': [1, 1, 1],
        'padding': "VALID",
        'data_format': "NWC",
        'name': "pool_case_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: NWC, float64, SAME padding, ksize=2, strides=2
    input_dict = {
        'input': np.random.randn(2, 16, 4).astype(np.float64),
        'ksize': [2],
        'strides': [2],
        'padding': "SAME",
        'data_format': "NWC",
        'name': "pool_case_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: NWC, positive integers, ksize=[1, 2, 1]
    input_dict = {
        'input': np.arange(24).reshape(2, 4, 3).astype(np.float32),
        'ksize': [1, 2, 1],
        'strides': [1, 1, 1],
        'padding': "VALID",
        'data_format': "NWC",
        'name': "pool_case_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: NWC, large window size, SAME padding
    input_dict = {
        'input': np.random.uniform(-10, 10, (1, 12, 4)).astype(np.float32),
        'ksize': [4],
        'strides': [3],
        'padding': "SAME",
        'data_format': "NWC",
        'name': "pool_case_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: NWC, mixed positive and negative, ksize=[1, 3, 1], strides=[1, 2, 1]
    input_dict = {
        'input': np.random.randn(3, 10, 3).astype(np.float32),
        'ksize': [1, 3, 1],
        'strides': [1, 2, 1],
        'padding': "VALID",
        'data_format': "NWC",
        'name': "pool_case_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: NWC, zeros input, SAME padding
    input_dict = {
        'input': np.zeros((2, 8, 2), dtype=np.float32),
        'ksize': [2],
        'strides': [1],
        'padding': "SAME",
        'data_format': "NWC",
        'name': "pool_case_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: NWC, ones input, VALID padding
    input_dict = {
        'input': np.ones((1, 5, 5), dtype=np.float32),
        'ksize': [3],
        'strides': [1],
        'padding': "VALID",
        'data_format': "NWC",
        'name': "pool_case_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: NWC, float32, SAME padding, ksize=[1, 3, 1], strides=[1, 2, 1]
    input_dict = {
        'input': np.random.randn(2, 6, 4).astype(np.float32),
        'ksize': [1, 3, 1],
        'strides': [1, 2, 1],
        'padding': "SAME",
        'data_format': "NWC",
        'name': "pool_case_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.max_pool1d"] = tf_nn_max_pool1d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool2d_inputs():
    list_of_inputs = []
    
    # Input 1
    input_dict = {
        'input': np.random.randn(1, 4, 4, 1).astype(np.float32),
        'ksize': [2, 2],
        'strides': [2, 2],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'name': 'pool1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'input': np.random.randn(2, 8, 8, 3).astype(np.float32),
        'ksize': [3, 3],
        'strides': [1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'name': 'pool2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'input': np.random.randn(1, 16, 16, 3).astype(np.float32),
        'ksize': [2, 2],
        'strides': [2, 2],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'name': 'pool3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'input': np.random.randn(4, 5, 5, 2).astype(np.float32),
        'ksize': [2, 2],
        'strides': [1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'name': 'pool4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'input': np.random.uniform(-10.0, 10.0, size=(1, 6, 6, 2)).astype(np.float32),
        'ksize': [2, 2],
        'strides': [2, 2],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'name': 'pool5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'input': np.random.randn(2, 10, 10, 4).astype(np.float32),
        'ksize': [4, 4],
        'strides': [2, 2],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'name': 'pool6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'input': np.random.randn(1, 28, 28, 1).astype(np.float32),
        'ksize': [3, 3],
        'strides': [2, 2],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'name': 'pool7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'input': np.random.randn(3, 14, 14, 8).astype(np.float32),
        'ksize': [2, 2],
        'strides': [2, 2],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'name': 'pool8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'input': np.random.randn(1, 1, 1, 1).astype(np.float32),
        'ksize': [1, 1],
        'strides': [1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'name': 'pool9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'input': np.random.randn(2, 32, 32, 4).astype(np.float32),
        'ksize': [2, 2],
        'strides': [1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'name': 'pool10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.max_pool2d"] = tf_nn_max_pool2d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool2d_inputs():
    list_of_inputs = []

    # Input 1: standard NHWC, float32, positive values
    input_val = np.random.uniform(0.0, 10.0, size=(1, 4, 4, 1)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'ksize': 2,
        'strides': 2,
        'padding': "VALID",
        'data_format': "NHWC",
        'name': "pool_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: NHWC, float32, with negative values, SAME padding
    input_val = np.random.uniform(-5.0, 5.0, size=(2, 8, 8, 3)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'ksize': 3,
        'strides': 1,
        'padding': "SAME",
        'data_format': "NHWC",
        'name': "pool_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: NHWC, float32, VALID padding, shape (1, 4, 4, 2)
    input_val = np.random.uniform(0.0, 1.0, size=(1, 4, 4, 2)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'ksize': 2,
        'strides': 2,
        'padding': "VALID",
        'data_format': "NHWC",
        'name': "pool_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large ksize and strides, float32, NHWC
    input_val = np.random.uniform(-10.0, 10.0, size=(1, 16, 16, 2)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'ksize': 4,
        'strides': 4,
        'padding': "SAME",
        'data_format': "NHWC",
        'name': "pool_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64 data type, NHWC, VALID
    input_val = np.random.uniform(0.0, 100.0, size=(1, 6, 6, 1)).astype(np.float64)
    input_dict = {
        'input': input_val,
        'ksize': 2,
        'strides': 1,
        'padding': "VALID",
        'data_format': "NHWC",
        'name': "pool_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, NHWC, SAME
    input_val = np.random.uniform(-1.0, 1.0, size=(2, 8, 8, 3)).astype(np.float64)
    input_dict = {
        'input': input_val,
        'ksize': 2,
        'strides': 2,
        'padding': "SAME",
        'data_format': "NHWC",
        'name': "pool_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Smallest spatial size (2x2), ksize=2, strides=1, NHWC
    input_val = np.random.uniform(0.0, 5.0, size=(1, 2, 2, 1)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'ksize': 2,
        'strides': 1,
        'padding': "VALID",
        'data_format': "NHWC",
        'name': "pool_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: ksize=1, strides=1, NHWC (identity pool)
    input_val = np.random.uniform(-10.0, 10.0, size=(1, 5, 5, 4)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'ksize': 1,
        'strides': 1,
        'padding': "SAME",
        'data_format': "NHWC",
        'name': "pool_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large batch size, NHWC, float32, VALID
    input_val = np.random.uniform(0.0, 1.0, size=(8, 10, 10, 3)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'ksize': 3,
        'strides': 2,
        'padding': "VALID",
        'data_format': "NHWC",
        'name': "pool_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: High channel count, float32, NHWC, SAME
    input_val = np.random.uniform(-0.5, 0.5, size=(1, 6, 6, 16)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'ksize': 2,
        'strides': 2,
        'padding': "SAME",
        'data_format': "NHWC",
        'name': "pool_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.max_pool2d_1"] = tf_nn_max_pool2d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool2d_inputs():
    list_of_inputs = []
    
    # Input 1
    input_dict = {
        "input": np.random.randn(1, 4, 4, 1).astype(np.float32),
        "ksize": [2, 2],
        "strides": 2,
        "padding": "VALID",
        "data_format": "NHWC",
        "name": "pool1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input_dict = {
        "input": np.random.randn(2, 8, 8, 3).astype(np.float32),
        "ksize": [3, 3],
        "strides": 1,
        "padding": "SAME",
        "data_format": "NHWC",
        "name": "pool2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input_dict = {
        "input": np.random.randn(1, 16, 16, 3).astype(np.float32),
        "ksize": [2, 2],
        "strides": 2,
        "padding": "VALID",
        "data_format": "NHWC",
        "name": "pool3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_dict = {
        "input": np.random.randn(4, 10, 10, 2).astype(np.float32),
        "ksize": [1, 2, 2, 1],
        "strides": 1,
        "padding": "SAME",
        "data_format": "NHWC",
        "name": "pool4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_dict = {
        "input": np.random.randn(1, 5, 5, 1).astype(np.float32),
        "ksize": [2, 2],
        "strides": 1,
        "padding": "VALID",
        "data_format": "NHWC",
        "name": "pool5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_dict = {
        "input": np.random.randn(2, 32, 32, 2).astype(np.float32),
        "ksize": [4, 4],
        "strides": 2,
        "padding": "SAME",
        "data_format": "NHWC",
        "name": "pool6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_dict = {
        "input": np.random.randint(-10, 10, size=(1, 6, 6, 1)).astype(np.float32),
        "ksize": [3, 3],
        "strides": 3,
        "padding": "VALID",
        "data_format": "NHWC",
        "name": "pool7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_dict = {
        "input": np.random.randn(1, 12, 12, 4).astype(np.float32),
        "ksize": [2, 2],
        "strides": 2,
        "padding": "SAME",
        "data_format": "NHWC",
        "name": "pool8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_dict = {
        "input": np.random.randn(3, 14, 14, 4).astype(np.float32),
        "ksize": [2, 2],
        "strides": 1,
        "padding": "VALID",
        "data_format": "NHWC",
        "name": "pool9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_dict = {
        "input": np.random.randn(1, 28, 28, 1).astype(np.float32),
        "ksize": [5, 5],
        "strides": 1,
        "padding": "SAME",
        "data_format": "NHWC",
        "name": "pool10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.max_pool2d_2"] = tf_nn_max_pool2d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool2d_inputs():
    list_of_inputs = []

    # Case 1: Standard pool, NHWC, VALID padding
    input_dict_1 = {
        'input': np.random.randn(1, 4, 4, 1).astype(np.float32),
        'ksize': 2,
        'strides': [2, 2],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'name': 'pool_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: SAME padding, NHWC, stride 1
    input_dict_2 = {
        'input': np.random.randn(2, 8, 8, 3).astype(np.float32),
        'ksize': 3,
        'strides': [1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'name': 'pool_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: NHWC format, VALID padding
    input_dict_3 = {
        'input': np.random.randn(1, 10, 10, 3).astype(np.float32),
        'ksize': 2,
        'strides': [2, 2],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'name': 'pool_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: Large ksize and strides, NHWC, VALID
    input_dict_4 = {
        'input': np.random.randn(4, 16, 16, 64).astype(np.float32),
        'ksize': 4,
        'strides': [4, 4],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'name': 'pool_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: 1x1 stride, SAME padding
    input_dict_5 = {
        'input': np.random.randn(1, 14, 14, 1).astype(np.float32),
        'ksize': 2,
        'strides': [1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'name': 'pool_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: ksize 1, stride 1 (effectively a no-op but valid)
    input_dict_6 = {
        'input': np.random.randn(3, 5, 5, 2).astype(np.float32),
        'ksize': 1,
        'strides': [1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'name': 'pool_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: NHWC format, larger input, SAME padding
    input_dict_7 = {
        'input': np.random.randn(2, 32, 32, 4).astype(np.float32),
        'ksize': 2,
        'strides': [1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'name': 'pool_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: Large strides, matching ksize
    input_dict_8 = {
        'input': np.random.randn(1, 25, 25, 1).astype(np.float32),
        'ksize': 5,
        'strides': [5, 5],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'name': 'pool_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: 3-channel input, typical image size, NHWC
    input_dict_9 = {
        'input': np.random.randn(2, 112, 112, 3).astype(np.float32),
        'ksize': 3,
        'strides': [2, 2],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'name': 'pool_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: 3-channel input, typical image size, NHWC
    input_dict_10 = {
        'input': np.random.randn(2, 112, 112, 3).astype(np.float32),
        'ksize': 3,
        'strides': [2, 2],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'name': 'pool_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.nn.max_pool2d_3"] = tf_nn_max_pool2d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool2d_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "input": np.random.randn(2, 4, 4, 3).astype(np.float32),
        "ksize": [2, 2],
        "strides": [2, 2],
        "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "data_format": "NHWC",
        "name": "pool1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "input": np.random.randn(1, 8, 8, 1).astype(np.float32),
        "ksize": [1, 3, 3, 1],
        "strides": [1, 1, 1, 1],
        "padding": [[0, 0], [1, 1], [1, 1], [0, 0]],
        "data_format": "NHWC",
        "name": "pool2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "input": np.random.randn(4, 16, 16, 2).astype(np.float32),
        "ksize": [2, 2],
        "strides": [2, 2],
        "padding": [[0, 0], [1, 1], [1, 1], [0, 0]],
        "data_format": "NHWC",
        "name": "pool3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "input": np.random.randint(-10, 10, size=(1, 5, 5, 2)).astype(np.float32),
        "ksize": [1, 2, 2, 1],
        "strides": [1, 2, 2, 1],
        "padding": [[0, 0], [0, 1], [0, 1], [0, 0]],
        "data_format": "NHWC",
        "name": "pool4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "input": np.random.randn(2, 10, 10, 3).astype(np.float32),
        "ksize": [3, 3],
        "strides": [1, 1],
        "padding": [[0, 0], [2, 2], [2, 2], [0, 0]],
        "data_format": "NHWC",
        "name": "pool5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "input": np.random.randn(5, 6, 6, 4).astype(np.float32),
        "ksize": [2, 2],
        "strides": [1, 1],
        "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "data_format": "NHWC",
        "name": "pool6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "input": np.random.randn(1, 4, 4, 2).astype(np.float32),
        "ksize": [1, 2, 2, 1],
        "strides": [1, 1, 1, 1],
        "padding": [[0, 0], [1, 1], [1, 1], [0, 0]],
        "data_format": "NHWC",
        "name": "pool7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "input": np.random.randn(3, 12, 12, 3).astype(np.float32),
        "ksize": [1, 4, 4, 1],
        "strides": [1, 2, 2, 1],
        "padding": [[0, 0], [1, 1], [1, 1], [0, 0]],
        "data_format": "NHWC",
        "name": "pool8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "input": np.random.randn(2, 14, 14, 1).astype(np.float32),
        "ksize": [3, 3],
        "strides": [2, 2],
        "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "data_format": "NHWC",
        "name": "pool9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "input": np.random.randn(1, 20, 20, 1).astype(np.float32),
        "ksize": [5, 5],
        "strides": [5, 5],
        "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "data_format": "NHWC",
        "name": "pool10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.max_pool2d_4"] = tf_nn_max_pool2d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool2d_inputs():
    list_of_inputs = []
    
    # Input 1
    input_dict = {
        'input': np.random.randn(1, 4, 4, 1).astype(np.float32),
        'ksize': 2,
        'strides': 1,
        'padding': [[0, 0], [1, 1], [1, 1], [0, 0]],
        'data_format': 'NHWC',
        'name': 'pool1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input_dict = {
        'input': np.random.randn(2, 8, 8, 3).astype(np.float32),
        'ksize': 3,
        'strides': 2,
        'padding': [[0, 0], [2, 2], [2, 2], [0, 0]],
        'data_format': 'NHWC',
        'name': 'pool2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input_dict = {
        'input': np.array([[[[-1.0], [-2.0]], [[-3.0], [-4.0]]]], dtype=np.float32),
        'ksize': 2,
        'strides': 1,
        'padding': [[0, 0], [0, 0], [0, 0], [0, 0]],
        'data_format': 'NHWC',
        'name': 'pool3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_dict = {
        'input': np.random.randn(1, 6, 6, 2).astype(np.float32),
        'ksize': 2,
        'strides': 2,
        'padding': [[0, 0], [1, 1], [1, 1], [0, 0]],
        'data_format': 'NHWC',
        'name': 'pool4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_dict = {
        'input': np.ones((4, 10, 10, 3), dtype=np.float32),
        'ksize': 4,
        'strides': 2,
        'padding': [[0, 0], [3, 3], [3, 3], [0, 0]],
        'data_format': 'NHWC',
        'name': 'pool5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_dict = {
        'input': np.random.randn(1, 12, 12, 1).astype(np.float32),
        'ksize': 3,
        'strides': 3,
        'padding': [[0, 0], [1, 1], [1, 1], [0, 0]],
        'data_format': 'NHWC',
        'name': 'pool6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_dict = {
        'input': np.random.randn(2, 5, 5, 4).astype(np.float32),
        'ksize': 1,
        'strides': 1,
        'padding': [[0, 0], [0, 0], [0, 0], [0, 0]],
        'data_format': 'NHWC',
        'name': 'pool7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_dict = {
        'input': np.random.randn(1, 3, 3, 1).astype(np.float64),
        'ksize': 2,
        'strides': 1,
        'padding': [[0, 0], [0, 0], [0, 0], [0, 0]],
        'data_format': 'NHWC',
        'name': 'pool8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_dict = {
        'input': np.random.randn(1, 16, 16, 2).astype(np.float32),
        'ksize': 4,
        'strides': 4,
        'padding': [[0, 0], [2, 2], [2, 2], [0, 0]],
        'data_format': 'NHWC',
        'name': 'pool9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_dict = {
        'input': np.zeros((1, 5, 5, 1), dtype=np.float32),
        'ksize': 3,
        'strides': 2,
        'padding': [[0, 0], [1, 1], [1, 1], [0, 0]],
        'data_format': 'NHWC',
        'name': 'pool10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.max_pool2d_5"] = tf_nn_max_pool2d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool2d_inputs():
    list_of_inputs = []
    
    # Input 1
    input_dict = {
        "input": np.random.randn(1, 4, 4, 1).astype(np.float32),
        "ksize": [2, 2],
        "strides": 2,
        "padding": [[0, 0], [1, 1], [1, 1], [0, 0]],
        "data_format": "NHWC",
        "name": "pool1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input_dict = {
        "input": np.random.randn(2, 8, 8, 3).astype(np.float32),
        "ksize": [3, 3],
        "strides": 1,
        "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "data_format": "NHWC",
        "name": "pool2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input_dict = {
        "input": np.random.randn(1, 16, 16, 3).astype(np.float32),
        "ksize": [2, 2],
        "strides": 2,
        "padding": [[0, 0], [1, 1], [1, 1], [0, 0]],
        "data_format": "NHWC",
        "name": "pool3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_dict = {
        "input": np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1),
        "ksize": [2, 2],
        "strides": 1,
        "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "data_format": "NHWC",
        "name": "pool4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_dict = {
        "input": np.ones((1, 5, 5, 2), dtype=np.float32),
        "ksize": [2, 2],
        "strides": 1,
        "padding": [[0, 0], [1, 1], [1, 1], [0, 0]],
        "data_format": "NHWC",
        "name": "pool5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_dict = {
        "input": (np.random.randn(1, 6, 6, 1) * 10).astype(np.float32),
        "ksize": [3, 3],
        "strides": 3,
        "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "data_format": "NHWC",
        "name": "pool6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_dict = {
        "input": np.random.uniform(-1, 1, (2, 10, 10, 4)).astype(np.float32),
        "ksize": [2, 2],
        "strides": 2,
        "padding": [[0, 0], [1, 1], [1, 1], [0, 0]],
        "data_format": "NHWC",
        "name": "pool7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_dict = {
        "input": np.zeros((1, 8, 8, 2), dtype=np.float32),
        "ksize": [2, 2],
        "strides": 2,
        "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "data_format": "NHWC",
        "name": "pool8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_dict = {
        "input": np.random.randn(4, 12, 12, 3).astype(np.float32),
        "ksize": [4, 4],
        "strides": 4,
        "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "data_format": "NHWC",
        "name": "pool9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_dict = {
        "input": np.random.randn(1, 14, 14, 1).astype(np.float32),
        "ksize": [2, 2],
        "strides": 2,
        "padding": [[0, 0], [1, 1], [1, 1], [0, 0]],
        "data_format": "NHWC",
        "name": "pool10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.max_pool2d_6"] = tf_nn_max_pool2d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool2d_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "input": np.random.randn(1, 4, 4, 1).astype(np.float32),
        "ksize": 2,
        "strides": [2, 2],
        "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "data_format": "NHWC",
        "name": "pool1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "input": np.random.randn(2, 6, 6, 3).astype(np.float32),
        "ksize": 3,
        "strides": [1, 1],
        "padding": [[0, 0], [1, 1], [1, 1], [0, 0]],
        "data_format": "NHWC",
        "name": "pool2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "input": np.random.randn(1, 8, 8, 3).astype(np.float32),
        "ksize": 2,
        "strides": [2, 2],
        "padding": [[0, 0], [1, 1], [1, 1], [0, 0]],
        "data_format": "NHWC",
        "name": "pool3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "input": np.random.randn(1, 4, 4, 2).astype(np.float32),
        "ksize": 2,
        "strides": [1, 1, 1, 1],
        "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "data_format": "NHWC",
        "name": "pool4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "input": np.random.randn(1, 5, 5, 2).astype(np.float64),
        "ksize": 2,
        "strides": [1, 2, 2, 1],
        "padding": [[0, 0], [1, 1], [1, 1], [0, 0]],
        "data_format": "NHWC",
        "name": "pool5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "input": np.random.randn(1, 4, 4, 1).astype(np.float16),
        "ksize": 2,
        "strides": [1, 1],
        "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "data_format": "NHWC",
        "name": "pool6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "input": -np.abs(np.random.randn(2, 5, 5, 2).astype(np.float32)),
        "ksize": 2,
        "strides": [2, 2],
        "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "data_format": "NHWC",
        "name": "pool7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "input": np.random.randn(1, 10, 10, 3).astype(np.float32),
        "ksize": 3,
        "strides": [2, 2],
        "padding": [[0, 0], [1, 1], [1, 1], [0, 0]],
        "data_format": "NHWC",
        "name": "pool8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "input": np.random.randn(2, 2, 2, 2).astype(np.float32),
        "ksize": 1,
        "strides": [1, 1],
        "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "data_format": "NHWC",
        "name": "pool9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "input": np.random.randn(3, 10, 10, 4).astype(np.float32),
        "ksize": 4,
        "strides": [3, 3],
        "padding": [[0, 0], [1, 2], [2, 1], [0, 0]],
        "data_format": "NHWC",
        "name": "pool10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.max_pool2d_7"] = tf_nn_max_pool2d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool3d_inputs():
    list_of_inputs = []

    # Input 1, valid, 5-D tensor, NDHWC format, float32
    input_dict = {
        "input": np.random.randn(1, 3, 3, 3, 1).astype(np.float32),
        "ksize": [1, 2, 2, 2, 1],
        "strides": [1, 1, 1, 1, 1],
        "padding": "VALID",
        "data_format": "NDHWC",
        "name": "pool_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid, SAME padding, NDHWC format, float32
    input_dict = {
        "input": np.random.randn(2, 4, 4, 4, 3).astype(np.float32),
        "ksize": [1, 2, 2, 2, 1],
        "strides": [1, 2, 2, 2, 1],
        "padding": "SAME",
        "data_format": "NDHWC",
        "name": "pool_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid, NDHWC format, VALID padding
    input_dict = {
        "input": np.random.randn(1, 5, 5, 5, 2).astype(np.float32),
        "ksize": [1, 3, 3, 3, 1],
        "strides": [1, 2, 2, 2, 1],
        "padding": "VALID",
        "data_format": "NDHWC",
        "name": "pool_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid, float32, 3-element ksize and strides list
    input_dict = {
        "input": np.random.randn(2, 2, 3, 3, 2).astype(np.float32),
        "ksize": [2, 2, 2],
        "strides": [1, 1, 1],
        "padding": "SAME",
        "data_format": "NDHWC",
        "name": "pool_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid, with larger dimension size, NDHWC
    input_dict = {
        "input": np.random.randn(1, 8, 8, 8, 1).astype(np.float32),
        "ksize": [1, 3, 3, 3, 1],
        "strides": [1, 2, 2, 2, 1],
        "padding": "SAME",
        "data_format": "NDHWC",
        "name": "pool_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid, negative values, 1-element ksize and strides list
    input_dict = {
        "input": -np.random.rand(1, 4, 4, 4, 4).astype(np.float32),
        "ksize": [1],
        "strides": [1],
        "padding": "VALID",
        "data_format": "NDHWC",
        "name": "pool_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid, NDHWC format, SAME padding, float32
    input_dict = {
        "input": np.random.randn(1, 10, 10, 10, 3).astype(np.float32),
        "ksize": [3, 3, 3],
        "strides": [2, 2, 2],
        "padding": "SAME",
        "data_format": "NDHWC",
        "name": "pool_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid, strides larger than window size
    input_dict = {
        "input": np.random.randn(3, 2, 2, 2, 1).astype(np.float32),
        "ksize": [2, 2, 2],
        "strides": [3, 3, 3],
        "padding": "VALID",
        "data_format": "NDHWC",
        "name": "pool_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid, minimum spatial dimension 1
    input_dict = {
        "input": np.random.randn(1, 1, 1, 1, 1).astype(np.float32),
        "ksize": [1, 1, 1, 1, 1],
        "strides": [1, 1, 1, 1, 1],
        "padding": "SAME",
        "data_format": "NDHWC",
        "name": "pool_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid, float32, NDHWC format
    input_dict = {
        "input": np.random.randn(2, 2, 2, 2, 5).astype(np.float32),
        "ksize": [1, 2, 2, 2, 1],
        "strides": [1, 1, 1, 1, 1],
        "padding": "VALID",
        "data_format": "NDHWC",
        "name": "pool_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.max_pool3d"] = tf_nn_max_pool3d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool_with_argmax_inputs():
    list_of_inputs = []
    
    # Case 1
    input_dict = {
        'input': np.random.randn(1, 4, 4, 1).astype(np.float32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'output_dtype': np.int64,
        'include_batch_in_index': False,
        'name': 'max_pool_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2
    input_dict = {
        'input': np.random.randint(-10, 10, size=(2, 3, 3, 2)).astype(np.int32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'output_dtype': np.int32,
        'include_batch_in_index': True,
        'name': 'max_pool_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3
    input_dict = {
        'input': np.random.randn(1, 8, 8, 3).astype(np.float64),
        'ksize': [1, 3, 3, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'output_dtype': np.int64,
        'include_batch_in_index': True,
        'name': 'max_pool_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4
    input_dict = {
        'input': np.random.randn(3, 5, 5, 4).astype(np.float16),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'output_dtype': np.int32,
        'include_batch_in_index': False,
        'name': 'max_pool_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5
    input_dict = {
        'input': np.random.randint(-5, 5, size=(1, 2, 2, 1)).astype(np.int16),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'output_dtype': np.int64,
        'include_batch_in_index': False,
        'name': 'max_pool_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6
    input_dict = {
        'input': np.random.randn(4, 10, 10, 3).astype(np.float32),
        'ksize': [1, 4, 4, 1],
        'strides': [1, 3, 3, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'output_dtype': np.int32,
        'include_batch_in_index': True,
        'name': 'max_pool_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7
    input_dict = {
        'input': np.random.randint(-128, 127, size=(2, 6, 6, 2)).astype(np.int8),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'output_dtype': np.int64,
        'include_batch_in_index': False,
        'name': 'max_pool_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8
    input_dict = {
        'input': np.random.randint(0, 255, size=(1, 7, 7, 1)).astype(np.uint8),
        'ksize': [1, 3, 3, 1],
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'output_dtype': np.int32,
        'include_batch_in_index': True,
        'name': 'max_pool_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9
    input_dict = {
        'input': np.zeros((5, 4, 4, 8)).astype(np.float32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'output_dtype': np.int64,
        'include_batch_in_index': True,
        'name': 'max_pool_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10
    input_dict = {
        'input': np.random.randint(-1000, 1000, size=(2, 2, 2, 2)).astype(np.int64),
        'ksize': [1, 1, 1, 1],
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'output_dtype': np.int32,
        'include_batch_in_index': False,
        'name': 'max_pool_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.max_pool_with_argmax"] = tf_nn_max_pool_with_argmax_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_moments_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis 0, keepdims False
    input_dict = {
        "x": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "axes": [0],
        "shift": np.array(0.0, dtype=np.float32),
        "keepdims": False,
        "name": "moments_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, axis 0, keepdims True, negative values
    input_dict = {
        "x": np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32),
        "axes": [0],
        "shift": np.array(1.0, dtype=np.float32),
        "keepdims": True,
        "name": "moments_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, axis 1, keepdims False
    input_dict = {
        "x": np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float64),
        "axes": [1],
        "shift": np.array(0.5, dtype=np.float64),
        "keepdims": False,
        "name": "moments_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array, axes [0, 1], keepdims True
    input_dict = {
        "x": np.arange(12, dtype=np.float32).reshape(2, 3, 2),
        "axes": [0, 1],
        "shift": np.array(0.0, dtype=np.float32),
        "keepdims": True,
        "name": "moments_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float32 array (e.g. convolutional shape), global normalization axes [0, 1, 2]
    input_dict = {
        "x": np.random.uniform(-10.0, 10.0, size=(2, 4, 4, 3)).astype(np.float32),
        "axes": [0, 1, 2],
        "shift": np.array(-1.0, dtype=np.float32),
        "keepdims": False,
        "name": "moments_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D float64 array, axis 0, keepdims True
    input_dict = {
        "x": np.array([100.0, 200.0, 300.0], dtype=np.float64),
        "axes": [0],
        "shift": np.array(100.0, dtype=np.float64),
        "keepdims": True,
        "name": "moments_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32 array, axis [2], keepdims False
    input_dict = {
        "x": np.ones((2, 2, 2), dtype=np.float32) * -5.0,
        "axes": [2],
        "shift": np.array(0.0, dtype=np.float32),
        "keepdims": False,
        "name": "moments_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float32 array, axes [1, 2], keepdims True
    input_dict = {
        "x": np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1),
        "axes": [1, 2],
        "shift": np.array(2.0, dtype=np.float32),
        "keepdims": True,
        "name": "moments_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32 array, axes [0, 1], keepdims False
    input_dict = {
        "x": np.array([[1000.0, 2000.0], [3000.0, 4000.0]], dtype=np.float32),
        "axes": [0, 1],
        "shift": np.array(1000.0, dtype=np.float32),
        "keepdims": False,
        "name": "moments_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D float32 array, axes [0, 2, 4], keepdims True
    input_dict = {
        "x": np.random.normal(size=(2, 2, 2, 2, 2)).astype(np.float32),
        "axes": [0, 2, 4],
        "shift": np.array(0.0, dtype=np.float32),
        "keepdims": True,
        "name": "moments_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.moments"] = tf_nn_moments_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_pool_inputs():
    list_of_inputs = []

    # Input 1: N=1, NWC, MAX, VALID
    input_dict = {
        'input': np.random.randn(2, 10, 3).astype(np.float32),
        'window_shape': [3],
        'pooling_type': 'MAX',
        'strides': [1],
        'padding': 'VALID',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'pool_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: N=1, NWC, AVG, SAME
    input_dict = {
        'input': np.random.randn(2, 10, 3).astype(np.float32),
        'window_shape': [2],
        'pooling_type': 'AVG',
        'strides': [2],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'pool_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: N=2, NHWC, MAX, SAME
    input_dict = {
        'input': np.random.randn(2, 8, 8, 3).astype(np.float32),
        'window_shape': [2, 2],
        'pooling_type': 'MAX',
        'strides': [2, 2],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'pool_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: N=2, NHWC, AVG, VALID, dilation > 1
    input_dict = {
        'input': np.random.randn(2, 8, 8, 3).astype(np.float32),
        'window_shape': [3, 3],
        'pooling_type': 'AVG',
        'strides': [1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [2, 2],
        'name': 'pool_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: N=3, NDHWC, MAX, SAME
    input_dict = {
        'input': np.random.randn(2, 4, 4, 4, 3).astype(np.float32),
        'window_shape': [2, 2, 2],
        'pooling_type': 'MAX',
        'strides': [1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NDHWC',
        'dilations': [1, 1, 1],
        'name': 'pool_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: N=3, NDHWC, AVG, VALID
    input_dict = {
        'input': np.random.randn(2, 4, 4, 4, 3).astype(np.float32),
        'window_shape': [2, 2, 2],
        'pooling_type': 'AVG',
        'strides': [2, 2, 2],
        'padding': 'VALID',
        'data_format': 'NDHWC',
        'dilations': [1, 1, 1],
        'name': 'pool_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: N=2, NHWC, asymmetric strides
    input_dict = {
        'input': np.random.randn(1, 10, 12, 4).astype(np.float32),
        'window_shape': [3, 2],
        'pooling_type': 'MAX',
        'strides': [2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'pool_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: N=2, NHWC, dilation > 1, VALID (SAME not supported for dilation > 1)
    input_dict = {
        'input': np.random.randn(1, 10, 10, 2).astype(np.float32),
        'window_shape': [3, 3],
        'pooling_type': 'MAX',
        'strides': [1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [2, 2],
        'name': 'pool_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: N=1, NWC, MAX, negative values
    input_dict = {
        'input': np.random.uniform(-10.0, -1.0, (4, 15, 2)).astype(np.float32),
        'window_shape': [4],
        'pooling_type': 'MAX',
        'strides': [3],
        'padding': 'SAME',
        'data_format': 'NWC',
        'dilations': [1],
        'name': 'pool_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: N=2, NHWC, AVG, VALID, large input
    input_dict = {
        'input': np.random.randn(1, 16, 16, 1).astype(np.float32),
        'window_shape': [4, 4],
        'pooling_type': 'AVG',
        'strides': [1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [3, 3],
        'name': 'pool_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.pool"] = tf_nn_pool_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_nn_sparse_softmax_cross_entropy_with_logits_inputs():
    list_of_inputs = []
    
    # Input 1: 1D labels, 2D logits, float32, int32
    labels = np.array([0, 2, 1], dtype=np.int32)
    logits = np.array([[2.0, -1.0, 0.5],
                       [0.0, 1.0, 3.0],
                       [-1.0, 2.0, -2.0]], dtype=np.float32)
    name = "loss_1"
    list_of_inputs.append({"labels": labels, "logits": logits, "name": name})

    # Input 2: 2D labels, 3D logits, float64, int64
    labels = np.array([[0, 1], [2, 0]], dtype=np.int64)
    logits = np.array([[[1.5, -0.5, 2.0], [0.0, 2.5, -1.0]],
                       [[-2.0, 0.5, 1.0], [3.0, 1.0, 0.0]]], dtype=np.float64)
    name = "loss_2"
    list_of_inputs.append({"labels": labels, "logits": logits, "name": name})

    # Input 3: 1D labels, 2D logits, float16, int32
    labels = np.array([1], dtype=np.int32)
    logits = np.array([[10.0, -10.0]], dtype=np.float16)
    name = "loss_3"
    list_of_inputs.append({"labels": labels, "logits": logits, "name": name})

    # Input 4: Larger shapes, float32, negative values
    labels = np.array([4, 3, 2, 1, 0], dtype=np.int32)
    logits = np.random.uniform(-5.0, 5.0, (5, 5)).astype(np.float32)
    name = "loss_4"
    list_of_inputs.append({"labels": labels, "logits": logits, "name": name})

    # Input 5: Higher dimensions (3D labels, 4D logits), float32, int32
    labels = np.random.randint(0, 5, size=(2, 3, 4)).astype(np.int32)
    logits = np.random.normal(0.0, 1.0, size=(2, 3, 4, 5)).astype(np.float32)
    name = "loss_5"
    list_of_inputs.append({"labels": labels, "logits": logits, "name": name})

    # Input 6: Minimal shapes, float32, int64
    labels = np.array([[1]], dtype=np.int64)
    logits = np.array([[[0.1, 0.9]]], dtype=np.float32)
    name = "loss_6"
    list_of_inputs.append({"labels": labels, "logits": logits, "name": name})

    # Input 7: Larger logits classes, float64, int32
    labels = np.array([0, 9], dtype=np.int32)
    logits = np.zeros((2, 10), dtype=np.float64)
    name = "loss_7"
    list_of_inputs.append({"labels": labels, "logits": logits, "name": name})

    # Input 8: Multi-dim (2D labels, 3D logits), float16, int64
    labels = np.array([[0, 2], [1, 1], [2, 0]], dtype=np.int64)
    logits = np.ones((3, 2, 3), dtype=np.float16)
    name = "loss_8"
    list_of_inputs.append({"labels": labels, "logits": logits, "name": name})

    # Input 9: Large number of classes (100 classes), 1D labels, float32
    labels = np.array([42, 99], dtype=np.int32)
    logits = np.random.standard_normal((2, 100)).astype(np.float32)
    name = "loss_9"
    list_of_inputs.append({"labels": labels, "logits": logits, "name": name})

    # Input 10: Higher dimension 3D, float64, int64
    labels = np.random.randint(0, 2, size=(2, 2, 2)).astype(np.int64)
    logits = np.random.uniform(-10.0, 10.0, size=(2, 2, 2, 2)).astype(np.float64)
    name = "loss_10"
    list_of_inputs.append({"labels": labels, "logits": logits, "name": name})

    return list_of_inputs

generated_inputs["tf.nn.sparse_softmax_cross_entropy_with_logits"] = tf_nn_sparse_softmax_cross_entropy_with_logits_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_quantization_fake_quant_with_min_max_args_inputs():
    list_of_inputs = []
    
    # Input 1
    list_of_inputs.append({
        'inputs': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        'min': -5.0,
        'max': 5.0,
        'num_bits': 8,
        'narrow_range': False,
        'name': "quant_1"
    })
    
    # Input 2
    list_of_inputs.append({
        'inputs': np.array([[-10.0, -5.0], [0.0, 5.0]], dtype=np.float32),
        'min': -10.0,
        'max': 10.0,
        'num_bits': 16,
        'narrow_range': True,
        'name': "quant_2"
    })
    
    # Input 3
    list_of_inputs.append({
        'inputs': np.array([[[1.5, 2.5], [3.5, 4.5]], [[5.5, 6.5], [7.5, 8.5]]], dtype=np.float32),
        'min': 0.0,
        'max': 9.0,
        'num_bits': 4,
        'narrow_range': False,
        'name': "quant_3"
    })
    
    # Input 4
    list_of_inputs.append({
        'inputs': np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        'min': -2.0,
        'max': 2.0,
        'num_bits': 2,
        'narrow_range': True,
        'name': "quant_4"
    })
    
    # Input 5
    list_of_inputs.append({
        'inputs': np.array([[-15.0, -12.0, -9.0], [9.0, 12.0, 15.0]], dtype=np.float32),
        'min': -12.0,
        'max': 12.0,
        'num_bits': 12,
        'narrow_range': False,
        'name': "quant_5"
    })
    
    # Input 6
    list_of_inputs.append({
        'inputs': np.array([-0.5, 0.5], dtype=np.float32),
        'min': -1.0,
        'max': 1.0,
        'num_bits': 8,
        'narrow_range': True,
        'name': "quant_6"
    })
    
    # Input 7
    list_of_inputs.append({
        'inputs': np.array([[-45.0, -25.0], [25.0, 45.0]], dtype=np.float32),
        'min': -50.0,
        'max': 50.0,
        'num_bits': 15,
        'narrow_range': False,
        'name': "quant_7"
    })
    
    # Input 8
    list_of_inputs.append({
        'inputs': np.array([0.1, 0.2, 0.3], dtype=np.float32),
        'min': 0.0,
        'max': 1.0,
        'num_bits': 8,
        'narrow_range': True,
        'name': "quant_8"
    })
    
    # Input 9
    list_of_inputs.append({
        'inputs': np.array([-10.0, -8.0, -6.0], dtype=np.float32),
        'min': -12.0,
        'max': -2.0,
        'num_bits': 10,
        'narrow_range': False,
        'name': "quant_9"
    })
    
    # Input 10
    list_of_inputs.append({
        'inputs': np.zeros((3, 3), dtype=np.float32),
        'min': -6.0,
        'max': 6.0,
        'num_bits': 8,
        'narrow_range': False,
        'name': "quant_10"
    })
    
    return list_of_inputs

generated_inputs["tf.quantization.fake_quant_with_min_max_args"] = tf_quantization_fake_quant_with_min_max_args_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_quantization_fake_quant_with_min_max_args_gradient_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor, standard defaults
    gradients_1 = np.array([0.1, -0.2, 0.5, 1.2], dtype=np.float32)
    inputs_1 = np.array([-1.5, 2.0, 5.5, -7.0], dtype=np.float32)
    input_dict_1 = {
        "gradients": gradients_1,
        "inputs": inputs_1,
        "min": -6.0,
        "max": 6.0,
        "num_bits": 8,
        "narrow_range": False,
        "name": "fake_quant_grad_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D tensor, narrow range, 4 bits
    gradients_2 = np.ones((2, 3), dtype=np.float32) * 0.5
    inputs_2 = np.array([[-2.0, 0.0, 2.0], [-4.0, 1.0, 3.0]], dtype=np.float32)
    input_dict_2 = {
        "gradients": gradients_2,
        "inputs": inputs_2,
        "min": -3.0,
        "max": 3.0,
        "num_bits": 4,
        "narrow_range": True,
        "name": "fake_quant_grad_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D tensor, larger bits (16), asymmetric min/max
    gradients_3 = np.random.uniform(-1.0, 1.0, size=(2, 2, 2)).astype(np.float32)
    inputs_3 = np.random.uniform(-10.0, 10.0, size=(2, 2, 2)).astype(np.float32)
    input_dict_3 = {
        "gradients": gradients_3,
        "inputs": inputs_3,
        "min": -10.0,
        "max": 5.0,
        "num_bits": 16,
        "narrow_range": False,
        "name": "fake_quant_grad_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 4D tensor, narrow range True, small range
    gradients_4 = np.zeros((1, 2, 2, 1), dtype=np.float32)
    inputs_4 = np.array([[[[0.1], [0.2]], [[-0.1], [-0.2]]]], dtype=np.float32)
    input_dict_4 = {
        "gradients": gradients_4,
        "inputs": inputs_4,
        "min": -1.0,
        "max": 1.0,
        "num_bits": 8,
        "narrow_range": True,
        "name": "fake_quant_grad_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 1D tensor, low bit count (2 bits)
    gradients_5 = np.array([-0.5, 0.5], dtype=np.float32)
    inputs_5 = np.array([-2.5, 2.5], dtype=np.float32)
    input_dict_5 = {
        "gradients": gradients_5,
        "inputs": inputs_5,
        "min": -2.0,
        "max": 2.0,
        "num_bits": 2,
        "narrow_range": False,
        "name": "fake_quant_grad_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 2D tensor, positive values only
    gradients_6 = np.random.exponential(1.0, size=(3, 3)).astype(np.float32)
    inputs_6 = np.random.exponential(2.0, size=(3, 3)).astype(np.float32)
    input_dict_6 = {
        "gradients": gradients_6,
        "inputs": inputs_6,
        "min": 0.0,
        "max": 10.0,
        "num_bits": 8,
        "narrow_range": False,
        "name": "fake_quant_grad_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 3D tensor, negative values only
    gradients_7 = -np.random.exponential(1.0, size=(2, 3, 2)).astype(np.float32)
    inputs_7 = -np.random.exponential(2.0, size=(2, 3, 2)).astype(np.float32)
    input_dict_7 = {
        "gradients": gradients_7,
        "inputs": inputs_7,
        "min": -8.0,
        "max": 0.0,
        "num_bits": 8,
        "narrow_range": True,
        "name": "fake_quant_grad_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: High dimensional (5D) tensor
    gradients_8 = np.ones((1, 2, 2, 2, 1), dtype=np.float32) * -0.1
    inputs_8 = np.ones((1, 2, 2, 2, 1), dtype=np.float32) * 1.5
    input_dict_8 = {
        "gradients": gradients_8,
        "inputs": inputs_8,
        "min": -2.5,
        "max": 2.5,
        "num_bits": 12,
        "narrow_range": False,
        "name": "fake_quant_grad_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: 1D tensor, very narrow min/max range
    gradients_9 = np.array([10.0, -10.0, 5.0], dtype=np.float32)
    inputs_9 = np.array([0.001, -0.002, 0.005], dtype=np.float32)
    input_dict_9 = {
        "gradients": gradients_9,
        "inputs": inputs_9,
        "min": -0.01,
        "max": 0.01,
        "num_bits": 8,
        "narrow_range": False,
        "name": "fake_quant_grad_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: 2D tensor, high quantization bit width (15)
    gradients_10 = np.random.normal(0.0, 1.0, size=(4, 4)).astype(np.float32)
    inputs_10 = np.random.normal(0.0, 5.0, size=(4, 4)).astype(np.float32)
    input_dict_10 = {
        "gradients": gradients_10,
        "inputs": inputs_10,
        "min": -15.0,
        "max": 15.0,
        "num_bits": 15,
        "narrow_range": True,
        "name": "fake_quant_grad_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.quantization.fake_quant_with_min_max_args_gradient"] = tf_quantization_fake_quant_with_min_max_args_gradient_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_random_experimental_stateless_split_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'seed': np.array([1, 2], dtype=np.int32),
        'num': 2,
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'seed': np.array([42, 100], dtype=np.int32),
        'num': 3,
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'seed': np.array([-1, -2], dtype=np.int32),
        'num': 5,
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'seed': np.array([0, 0], dtype=np.int32),
        'num': 10,
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'seed': np.array([2147483647, -2147483648], dtype=np.int32),
        'num': 1,
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'seed': np.array([10, 20], dtype=np.int64),
        'num': 4,
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'seed': np.array([-999, 999], dtype=np.int64),
        'num': 2,
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'seed': np.array([123456789, 987654321], dtype=np.int64),
        'num': 8,
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'seed': np.array([1, 1], dtype=np.int32),
        'num': 6,
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'seed': np.array([987654, 3210], dtype=np.int32),
        'num': 12,
        'alg': 'auto_select'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random.experimental.stateless_split"] = tf_random_experimental_stateless_split_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_random_learned_unigram_candidate_sampler_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'true_classes': np.array([[1, 2]], dtype=np.int64),
        'num_true': 2,
        'num_sampled': 5,
        'unique': True,
        'range_max': 10,
        'seed': 42,
        'name': "sampler_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'true_classes': np.array([[0], [4], [3]], dtype=np.int64),
        'num_true': 1,
        'num_sampled': 3,
        'unique': False,
        'range_max': 5,
        'seed': 10,
        'name': "sampler_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'true_classes': np.array([[10, 20, 30], [5, 15, 25]], dtype=np.int64),
        'num_true': 3,
        'num_sampled': 10,
        'unique': True,
        'range_max': 100,
        'seed': 123,
        'name': "sampler_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'true_classes': np.array([[0, 1, 2, 3, 4]], dtype=np.int64),
        'num_true': 5,
        'num_sampled': 2,
        'unique': False,
        'range_max': 10,
        'seed': 0,
        'name': "sampler_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'true_classes': np.array([[0], [1]], dtype=np.int64),
        'num_true': 1,
        'num_sampled': 1,
        'unique': True,
        'range_max': 2,
        'seed': 99,
        'name': "sampler_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'true_classes': np.array([[5, 5], [3, 3]], dtype=np.int64),
        'num_true': 2,
        'num_sampled': 4,
        'unique': False,
        'range_max': 6,
        'seed': 7,
        'name': "sampler_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'true_classes': np.array([[99], [50], [0], [25]], dtype=np.int64),
        'num_true': 1,
        'num_sampled': 20,
        'unique': True,
        'range_max': 1000,
        'seed': 777,
        'name': "sampler_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'true_classes': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64),
        'num_true': 3,
        'num_sampled': 5,
        'unique': True,
        'range_max': 15,
        'seed': 888,
        'name': "sampler_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'true_classes': np.array([[0, 0], [0, 0]], dtype=np.int64),
        'num_true': 2,
        'num_sampled': 1,
        'unique': False,
        'range_max': 1,
        'seed': 1,
        'name': "sampler_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'true_classes': np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64),
        'num_true': 6,
        'num_sampled': 3,
        'unique': True,
        'range_max': 7,
        'seed': 55,
        'name': "sampler_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random.learned_unigram_candidate_sampler"] = tf_random_learned_unigram_candidate_sampler_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_random_stateless_categorical_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'logits': np.array([[0.1, 0.9]], dtype=np.float32),
        'num_samples': 5,
        'seed': np.array([7, 17], dtype=np.int32),
        'dtype': np.int64,
        'name': "stateless_cat_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'logits': np.array([[1.0, -2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        'num_samples': 10,
        'seed': np.array([42, 43], dtype=np.int32),
        'dtype': np.int32,
        'name': "stateless_cat_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'logits': np.random.normal(size=(5, 5)).astype(np.float32),
        'num_samples': 1,
        'seed': np.array([123, 456], dtype=np.int64),
        'dtype': np.int64,
        'name': "stateless_cat_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'logits': np.array([[-10.0, -10.0], [10.0, 10.0]], dtype=np.float64),
        'num_samples': 3,
        'seed': np.array([9, 8], dtype=np.int32),
        'dtype': np.int32,
        'name': "stateless_cat_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'logits': np.zeros((10, 2), dtype=np.float32),
        'num_samples': 2,
        'seed': np.array([100, 200], dtype=np.int32),
        'dtype': np.int64,
        'name': "stateless_cat_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'logits': np.log(np.ones((1, 100)) / 100.0).astype(np.float32),
        'num_samples': 50,
        'seed': np.array([1, 1], dtype=np.int32),
        'dtype': np.int32,
        'name': "stateless_cat_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'logits': np.array([[-1.0, 0.0, 1.0]], dtype=np.float32),
        'num_samples': 20,
        'seed': np.array([999, 999], dtype=np.int64),
        'dtype': np.int64,
        'name': "stateless_cat_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'logits': np.array([[100.0, 0.0], [0.0, 100.0]], dtype=np.float32),
        'num_samples': 15,
        'seed': np.array([777, 888], dtype=np.int32),
        'dtype': np.int32,
        'name': "stateless_cat_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'logits': np.array([[-0.5, -0.5, -0.5, -0.5]], dtype=np.float64),
        'num_samples': 8,
        'seed': np.array([3, 4], dtype=np.int64),
        'dtype': np.int64,
        'name': "stateless_cat_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'logits': np.random.uniform(-5.0, 5.0, size=(3, 10)).astype(np.float32),
        'num_samples': 4,
        'seed': np.array([5, 12], dtype=np.int32),
        'dtype': np.int32,
        'name': "stateless_cat_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random.stateless_categorical"] = tf_random_stateless_categorical_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_random_stateless_normal_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'shape': np.array([2, 3], dtype=np.int32),
        'seed': np.array([42, 43], dtype=np.int32),
        'mean': 0.0,
        'stddev': 1.0,
        'dtype': np.dtype('float32'),
        'name': "stateless_normal_1",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'shape': np.array([5], dtype=np.int32),
        'seed': np.array([1, 2], dtype=np.int32),
        'mean': 1.5,
        'stddev': 0.5,
        'dtype': np.dtype('float64'),
        'name': "stateless_normal_2",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'shape': np.array([2, 2, 2], dtype=np.int32),
        'seed': np.array([100, 200], dtype=np.int32),
        'mean': -1.0,
        'stddev': 2.0,
        'dtype': np.dtype('float16'),
        'name': "stateless_normal_3",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'shape': np.array([10], dtype=np.int32),
        'seed': np.array([0, 0], dtype=np.int32),
        'mean': 0.0,
        'stddev': 0.1,
        'dtype': np.dtype('float32'),
        'name': "stateless_normal_4",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'shape': np.array([3, 1, 4], dtype=np.int32),
        'seed': np.array([12, 34], dtype=np.int32),
        'mean': 100.0,
        'stddev': 15.0,
        'dtype': np.dtype('float64'),
        'name': "stateless_normal_5",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'shape': np.array([1], dtype=np.int32),
        'seed': np.array([7, 8], dtype=np.int32),
        'mean': -0.5,
        'stddev': 1.2,
        'dtype': np.dtype('float16'),
        'name': "stateless_normal_6",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'shape': np.array([4, 4], dtype=np.int32),
        'seed': np.array([999, 999], dtype=np.int32),
        'mean': 3.14,
        'stddev': 2.71,
        'dtype': np.dtype('float32'),
        'name': "stateless_normal_7",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'shape': np.array([2, 3, 4, 5], dtype=np.int32),
        'seed': np.array([7, 8], dtype=np.int32),
        'mean': 0.0,
        'stddev': 1.0,
        'dtype': np.dtype('float64'),
        'name': "stateless_normal_8",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'shape': np.array([100], dtype=np.int32),
        'seed': np.array([42, 42], dtype=np.int32),
        'mean': -10.0,
        'stddev': 0.01,
        'dtype': np.dtype('float32'),
        'name': "stateless_normal_9",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'shape': np.array([2, 2], dtype=np.int32),
        'seed': np.array([1111, 2222], dtype=np.int32),
        'mean': 0.001,
        'stddev': 0.002,
        'dtype': np.dtype('float16'),
        'name': "stateless_normal_10",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random.stateless_normal"] = tf_random_stateless_normal_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_random_stateless_truncated_normal_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'shape': np.array([2, 3], dtype=np.int32),
        'seed': np.array([1, 2], dtype=np.int32),
        'mean': 0.0,
        'stddev': 1.0,
        'dtype': np.float32,
        'name': "truncated_normal_1",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'shape': np.array([5], dtype=np.int64),
        'seed': np.array([42, 24], dtype=np.int64),
        'mean': -1.5,
        'stddev': 0.5,
        'dtype': np.float64,
        'name': "truncated_normal_2",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'shape': np.array([2, 2, 2], dtype=np.int32),
        'seed': np.array([7, 8], dtype=np.int32),
        'mean': 10.0,
        'stddev': 2.5,
        'dtype': np.float32,
        'name': "truncated_normal_3",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'shape': np.array([10], dtype=np.int32),
        'seed': np.array([0, 0], dtype=np.int32),
        'mean': 0.0,
        'stddev': 0.001,
        'dtype': np.float16,
        'name': "truncated_normal_4",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'shape': np.array([3, 1], dtype=np.int32),
        'seed': np.array([123, 456], dtype=np.int32),
        'mean': -100.0,
        'stddev': 50.0,
        'dtype': np.float64,
        'name': "truncated_normal_5",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'shape': np.array([1], dtype=np.int32),
        'seed': np.array([9, 9], dtype=np.int32),
        'mean': 0.5,
        'stddev': 0.1,
        'dtype': np.float32,
        'name': "truncated_normal_6",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'shape': np.array([1, 5, 1], dtype=np.int64),
        'seed': np.array([100, 200], dtype=np.int64),
        'mean': 3.14,
        'stddev': 1.59,
        'dtype': np.float32,
        'name': "truncated_normal_7",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'shape': np.array([4, 4], dtype=np.int32),
        'seed': np.array([11, 22], dtype=np.int32),
        'mean': -0.01,
        'stddev': 0.02,
        'dtype': np.float64,
        'name': "truncated_normal_8",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'shape': np.array([2, 3, 4], dtype=np.int32),
        'seed': np.array([1234, 5678], dtype=np.int32),
        'mean': 12.3,
        'stddev': 4.5,
        'dtype': np.float16,
        'name': "truncated_normal_9",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'shape': np.array([8], dtype=np.int64),
        'seed': np.array([999, 888], dtype=np.int64),
        'mean': -5.0,
        'stddev': 0.01,
        'dtype': np.float64,
        'name': "truncated_normal_10",
        'alg': "auto_select"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random.stateless_truncated_normal"] = tf_random_stateless_truncated_normal_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_AvgPool3D_inputs():
    list_of_inputs = []
    
    # 1. NDHWC, float32, SAME
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool1',
        'input': np.random.randn(1, 2, 2, 2, 1).astype(np.float32),
        'ksize': [1, 2, 2, 2, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2. NDHWC, float32, VALID
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool2',
        'input': np.random.randn(2, 4, 4, 4, 3).astype(np.float32),
        'ksize': [1, 2, 2, 2, 1],
        'strides': [1, 2, 2, 2, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3. NDHWC, float32, SAME, unit kernel
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool3',
        'input': np.random.randn(1, 3, 3, 3, 2).astype(np.float32),
        'ksize': [1, 1, 1, 1, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4. NDHWC, float32, VALID, larger kernel
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool4',
        'input': np.random.randn(2, 5, 5, 5, 1).astype(np.float32),
        'ksize': [1, 3, 3, 3, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5. NDHWC, float32, SAME, non-uniform strides
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool5',
        'input': np.random.randn(1, 4, 4, 4, 2).astype(np.float32),
        'ksize': [1, 2, 2, 2, 1],
        'strides': [1, 1, 2, 1, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6. NDHWC, float32, VALID, non-uniform kernel
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool6',
        'input': np.random.randn(3, 3, 3, 3, 3).astype(np.float32),
        'ksize': [1, 2, 1, 2, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7. NDHWC, float32, SAME, larger kernel and strides
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool7',
        'input': np.random.randn(2, 5, 5, 5, 2).astype(np.float32),
        'ksize': [1, 3, 3, 3, 1],
        'strides': [1, 2, 2, 2, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8. NDHWC, float32, VALID, multiple channels
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool8',
        'input': np.random.randn(1, 6, 6, 6, 4).astype(np.float32),
        'ksize': [1, 2, 2, 2, 1],
        'strides': [1, 2, 2, 2, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9. NDHWC, float32, VALID, minimal input shape
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool9',
        'input': np.random.randn(1, 1, 1, 1, 1).astype(np.float32),
        'ksize': [1, 1, 1, 1, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10. NDHWC, float32, SAME, asymmetric dimensions
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool10',
        'input': np.random.randn(2, 2, 3, 4, 2).astype(np.float32),
        'ksize': [1, 2, 1, 2, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.AvgPool3D"] = tf_raw_ops_AvgPool3D_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Bitcast_inputs():
    list_of_inputs = []

    # Input 1: Equal size, float32 to int32 conversion (1D array)
    list_of_inputs.append({
        'name': 'eq_size_i32_to_f32',
        'input': np.array([1, 2, -3], dtype=np.int32),
        'type': np.dtype('float32')
    })

    # Input 2: Larger to smaller, float32 (4 bytes) to uint8 (1 byte)
    list_of_inputs.append({
        'name': 'f32_to_u8',
        'input': np.array([1.0, -2.0], dtype=np.float32),
        'type': np.dtype('uint8')
    })

    # Input 3: Smaller to larger, uint8 (1 byte) to int32 (4 bytes)
    list_of_inputs.append({
        'name': 'u8_to_i32',
        'input': np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint8),
        'type': np.dtype('int32')
    })

    # Input 4: Equal size, complex64 (8 bytes) to int64 (8 bytes)
    list_of_inputs.append({
        'name': 'c64_to_i64',
        'input': np.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex64),
        'type': np.dtype('int64')
    })

    # Input 5: Equal size, int64 to float64 (1D array)
    list_of_inputs.append({
        'name': 'i64_to_f64',
        'input': np.array([1, -2, 3], dtype=np.int64),
        'type': np.dtype('float64')
    })

    # Input 6: Larger to smaller, float64 (8 bytes) to float32 (4 bytes) in 2D
    list_of_inputs.append({
        'name': 'f64_to_f32',
        'input': np.array([[1.0], [2.0]], dtype=np.float64),
        'type': np.dtype('float32')
    })

    # Input 7: Smaller to larger, float32 (4 bytes) to float64 (8 bytes)
    list_of_inputs.append({
        'name': 'f32_to_f64',
        'input': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        'type': np.dtype('float64')
    })

    # Input 8: Equal size, negative values, int16 to uint16
    list_of_inputs.append({
        'name': 'i16_to_u16',
        'input': np.array([-10, 0, 10], dtype=np.int16),
        'type': np.dtype('uint16')
    })

    # Input 9: Larger to smaller, int32 (4 bytes) to int16 (2 bytes)
    list_of_inputs.append({
        'name': 'i32_to_i16',
        'input': np.array([123456], dtype=np.int32),
        'type': np.dtype('int16')
    })

    # Input 10: Smaller to larger, int16 (2 bytes) to int32 (4 bytes)
    list_of_inputs.append({
        'name': 'i16_to_i32',
        'input': np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int16),
        'type': np.dtype('int32')
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.Bitcast"] = tf_raw_ops_Bitcast_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Bucketize_inputs():
    list_of_inputs = []

    # Input 1: Float32 1D input, simple boundaries
    list_of_inputs.append({
        'name': 'bucketize_1',
        'input': np.array([-1.5, 0.0, 1.5, 2.5, 10.0], dtype=np.float32),
        'boundaries': [0.0, 2.0]
    })

    # Input 2: Float32 2D input, matching the doc example
    list_of_inputs.append({
        'name': 'bucketize_2',
        'input': np.array([[-5, 10000], [150, 10], [5, 100]], dtype=np.float32),
        'boundaries': [0.0, 10.0, 100.0]
    })

    # Input 3: Int32 2D input, negative boundaries
    list_of_inputs.append({
        'name': 'bucketize_3',
        'input': np.array([[-10, 0], [10, 20]], dtype=np.int32),
        'boundaries': [-5.0, 5.0, 15.0]
    })

    # Input 4: Float64 3D input
    list_of_inputs.append({
        'name': 'bucketize_4',
        'input': np.array([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]], dtype=np.float64),
        'boundaries': [2.0, 4.0, 6.0, 8.0]
    })

    # Input 5: Int64 1D input, large values
    list_of_inputs.append({
        'name': 'bucketize_5',
        'input': np.array([100, 200, 300, 400], dtype=np.int64),
        'boundaries': [150.0, 250.0, 350.0]
    })

    # Input 6: Float32 1D input, empty boundaries
    list_of_inputs.append({
        'name': 'bucketize_6',
        'input': np.array([-1.0, 1.0], dtype=np.float32),
        'boundaries': []
    })

    # Input 7: Int32 Scalar (0D) input
    list_of_inputs.append({
        'name': 'bucketize_7',
        'input': np.array(5, dtype=np.int32),
        'boundaries': [0.0, 10.0]
    })

    # Input 8: Float64 2D input with many boundaries
    list_of_inputs.append({
        'name': 'bucketize_8',
        'input': np.array([[0.1, 0.9], [0.4, 0.6]], dtype=np.float64),
        'boundaries': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    })

    # Input 9: Int64 4D input
    list_of_inputs.append({
        'name': 'bucketize_9',
        'input': np.ones((2, 2, 2, 2), dtype=np.int64) * 10,
        'boundaries': [5.0, 15.0]
    })

    # Input 10: Float32 3D input with boundaries containing large ranges
    list_of_inputs.append({
        'name': 'bucketize_10',
        'input': np.array([[[1e-5, 1e5]]], dtype=np.float32),
        'boundaries': [1e-3, 1.0, 1e3]
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.Bucketize"] = tf_raw_ops_Bucketize_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Conv2D_inputs():
    list_of_inputs = []
    
    # Input 1: Standard float32 NHWC, valid padding, stride 1
    input_dict = {
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_1',
        'input': np.random.randn(2, 8, 8, 3).astype(np.float32),
        'filter': np.random.randn(3, 3, 3, 16).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Standard float32 NHWC, same padding, stride 2
    input_dict = {
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_2',
        'input': np.random.randn(1, 16, 16, 4).astype(np.float32),
        'filter': np.random.randn(5, 5, 4, 8).astype(np.float32),
        'strides': [1, 2, 2, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float64 NHWC, same padding, stride 1, dilation 2
    input_dict = {
        'use_cudnn_on_gpu': False,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 2, 2, 1],
        'name': 'conv_3',
        'input': np.random.randn(2, 10, 10, 3).astype(np.float64),
        'filter': np.random.randn(3, 3, 3, 4).astype(np.float64),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float16 NHWC, valid padding, stride 1
    input_dict = {
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_4',
        'input': np.random.randn(1, 14, 14, 1).astype(np.float16),
        'filter': np.random.randn(3, 3, 1, 2).astype(np.float16),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Int32 NHWC, same padding, stride 1
    input_dict = {
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_5',
        'input': np.random.randint(-10, 10, size=(1, 6, 6, 2)).astype(np.int32),
        'filter': np.random.randint(-5, 5, size=(2, 2, 2, 3)).astype(np.int32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: NHWC, Float32, same padding, stride 1
    input_dict = {
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_6',
        'input': np.random.randn(2, 12, 12, 3).astype(np.float32),
        'filter': np.random.randn(3, 3, 3, 8).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: NHWC, Float32, valid padding, stride 2
    input_dict = {
        'use_cudnn_on_gpu': False,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_7',
        'input': np.random.randn(4, 16, 16, 2).astype(np.float32),
        'filter': np.random.randn(3, 3, 2, 4).astype(np.float32),
        'strides': [1, 2, 2, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: EXPLICIT padding NHWC, Float32
    input_dict = {
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [0, 0, 1, 1, 2, 2, 0, 0],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_8',
        'input': np.random.randn(1, 8, 8, 3).astype(np.float32),
        'filter': np.random.randn(3, 3, 3, 5).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'EXPLICIT'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: EXPLICIT padding NHWC, Float32
    input_dict = {
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [0, 0, 1, 1, 2, 2, 0, 0],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_9',
        'input': np.random.randn(1, 8, 8, 3).astype(np.float32),
        'filter': np.random.randn(3, 3, 3, 5).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'EXPLICIT'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: NHWC, Float32, dilation 3, same padding, stride 1
    input_dict = {
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 3, 3, 1],
        'name': 'conv_10',
        'input': np.random.randn(1, 20, 20, 3).astype(np.float32),
        'filter': np.random.randn(3, 3, 3, 4).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Conv2D"] = tf_raw_ops_Conv2D_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DataFormatVecPermute_inputs():
    list_of_inputs = []

    # Case 1: n=4 vector, size n
    input_dict = {
        'src_format': 'NHWC',
        'dst_format': 'NCHW',
        'name': 'permute_1',
        'x': np.array([1, 2, 3, 4], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: n=4 vector, size n-2
    input_dict = {
        'src_format': 'NHWC',
        'dst_format': 'NCHW',
        'name': 'permute_2',
        'x': np.array([10, 20], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: n=5 tensor, shape (n, 2)
    input_dict = {
        'src_format': 'NDHWC',
        'dst_format': 'NCDHW',
        'name': 'permute_3',
        'x': np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: n=5 tensor, shape (n-2, 2)
    input_dict = {
        'src_format': 'NDHWC',
        'dst_format': 'NCDHW',
        'name': 'permute_4',
        'x': np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: n=4 vector, size n (with negative numbers)
    input_dict = {
        'src_format': 'NCHW',
        'dst_format': 'NHWC',
        'name': 'permute_5',
        'x': np.array([-1, -2, -3, -4], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: n=4 vector, size n-2
    input_dict = {
        'src_format': 'NCHW',
        'dst_format': 'NHWC',
        'name': 'permute_6',
        'x': np.array([-10, -20], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: n=5 tensor, shape (n, 2) with negative numbers
    input_dict = {
        'src_format': 'NCDHW',
        'dst_format': 'NDHWC',
        'name': 'permute_7',
        'x': np.array([[-1, 1], [-2, 2], [-3, 3], [-4, 4], [-5, 5]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: n=5 tensor, shape (n-2, 2)
    input_dict = {
        'src_format': 'NCDHW',
        'dst_format': 'NDHWC',
        'name': 'permute_8',
        'x': np.array([[-10, 10], [-20, 20], [-30, 30]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: n=4 tensor, shape (n, 2)
    input_dict = {
        'src_format': 'NHWC',
        'dst_format': 'NCHW',
        'name': 'permute_9',
        'x': np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: n=5 vector, size n
    input_dict = {
        'src_format': 'NDHWC',
        'dst_format': 'NCDHW',
        'name': 'permute_10',
        'x': np.array([100, 200, 300, 400, 500], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.DataFormatVecPermute"] = tf_raw_ops_DataFormatVecPermute_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_Dilation2D_inputs():
    list_of_inputs = []
    
    # Input 1
    input_val = np.random.randn(1, 3, 3, 1).astype(np.float32)
    filter_val = np.random.randn(2, 2, 1).astype(np.float32)
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "dilation_1"
    list_of_inputs.append({
        'input': input_val,
        'filter': filter_val,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 2
    input_val = np.random.randn(2, 4, 4, 3).astype(np.float64)
    filter_val = np.random.randn(3, 3, 3).astype(np.float64)
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"
    name = "dilation_2"
    list_of_inputs.append({
        'input': input_val,
        'filter': filter_val,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 3
    input_val = np.random.randint(-10, 10, size=(1, 5, 5, 2)).astype(np.int32)
    filter_val = np.random.randint(-10, 10, size=(2, 2, 2)).astype(np.int32)
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "dilation_3"
    list_of_inputs.append({
        'input': input_val,
        'filter': filter_val,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 4
    input_val = np.random.randn(1, 6, 6, 1).astype(np.float32)
    filter_val = np.random.randn(2, 2, 1).astype(np.float32)
    strides = [1, 1, 1, 1]
    rates = [1, 2, 2, 1]
    padding = "SAME"
    name = "dilation_4"
    list_of_inputs.append({
        'input': input_val,
        'filter': filter_val,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 5
    input_val = np.random.randint(-10, 10, size=(1, 3, 3, 1)).astype(np.int32)
    filter_val = np.random.randint(-10, 10, size=(1, 1, 1)).astype(np.int32)
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "dilation_5"
    list_of_inputs.append({
        'input': input_val,
        'filter': filter_val,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 6
    input_val = np.random.randn(2, 5, 5, 4).astype(np.float64)
    filter_val = np.random.randn(3, 3, 4).astype(np.float64)
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"
    name = "dilation_6"
    list_of_inputs.append({
        'input': input_val,
        'filter': filter_val,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 7
    input_val = np.random.randint(-50, 50, size=(1, 7, 7, 1)).astype(np.int32)
    filter_val = np.random.randint(-50, 50, size=(2, 2, 1)).astype(np.int32)
    strides = [1, 1, 1, 1]
    rates = [1, 3, 3, 1]
    padding = "VALID"
    name = "dilation_7"
    list_of_inputs.append({
        'input': input_val,
        'filter': filter_val,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 8
    input_val = np.random.randn(1, 4, 4, 2).astype(np.float32)
    filter_val = np.random.randn(2, 2, 2).astype(np.float32)
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"
    name = "dilation_8"
    list_of_inputs.append({
        'input': input_val,
        'filter': filter_val,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 9
    input_val = np.random.randn(1, 3, 3, 1).astype(np.float64)
    filter_val = np.random.randn(2, 2, 1).astype(np.float64)
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"
    name = "dilation_9"
    list_of_inputs.append({
        'input': input_val,
        'filter': filter_val,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 10
    input_val = np.random.randn(1, 10, 10, 1).astype(np.float32)
    filter_val = np.random.randn(3, 3, 1).astype(np.float32)
    strides = [1, 2, 2, 1]
    rates = [1, 2, 2, 1]
    padding = "VALID"
    name = "dilation_10"
    list_of_inputs.append({
        'input': input_val,
        'filter': filter_val,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.Dilation2D"] = tf_raw_ops_Dilation2D_inputs()

import numpy as np
import tensorflow as tf
import copy

def draw_bounding_boxes_inputs():
    list_of_inputs = []

    # Input 1: Basic Float32 image, 1 box
    images_1 = np.ones((1, 10, 10, 3), dtype=np.float32)
    boxes_1 = np.array([[[0.1, 0.1, 0.9, 0.9]]], dtype=np.float32)
    list_of_inputs.append({
        "name": "draw_1",
        "images": images_1,
        "boxes": boxes_1
    })

    # Input 2: Float16 (half) image, 2 boxes, batch of 2
    images_2 = np.zeros((2, 5, 5, 1), dtype=np.float16)
    boxes_2 = np.array([
        [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]],
        [[0.2, 0.2, 0.8, 0.8], [0.1, 0.1, 0.9, 0.9]]
    ], dtype=np.float32)
    list_of_inputs.append({
        "name": "draw_2",
        "images": images_2,
        "boxes": boxes_2
    })

    # Input 3: Zero boxes
    images_3 = np.random.rand(1, 20, 20, 4).astype(np.float32)
    boxes_3 = np.empty((1, 0, 4), dtype=np.float32)
    list_of_inputs.append({
        "name": "draw_empty_boxes",
        "images": images_3,
        "boxes": boxes_3
    })

    # Input 4: Boxes out of bounds (negative/larger than 1)
    images_4 = np.zeros((3, 8, 8, 3), dtype=np.float32)
    boxes_4 = np.array([
        [[-0.2, -0.2, 1.2, 1.2]],
        [[-0.5, 0.0, 1.5, 1.0]],
        [[0.0, -0.5, 1.0, 1.5]]
    ], dtype=np.float32)
    list_of_inputs.append({
        "name": "draw_out_of_bounds",
        "images": images_4,
        "boxes": boxes_4
    })

    # Input 5: Float16 image, multiple boxes
    images_5 = np.ones((1, 100, 100, 3), dtype=np.float16)
    boxes_5 = np.random.rand(1, 5, 4).astype(np.float32)
    list_of_inputs.append({
        "name": "draw_multiple_boxes",
        "images": images_5,
        "boxes": boxes_5
    })

    # Input 6: Zero size boxes (all 0.0)
    images_6 = np.random.rand(4, 4, 4, 3).astype(np.float32)
    boxes_6 = np.zeros((4, 1, 4), dtype=np.float32)
    list_of_inputs.append({
        "name": "draw_zero_boxes",
        "images": images_6,
        "boxes": boxes_6
    })

    # Input 7: Ones boxes (all 1.0) with valid channel depth (3)
    images_7 = np.random.rand(1, 10, 10, 3).astype(np.float32)
    boxes_7 = np.ones((1, 2, 4), dtype=np.float32)
    list_of_inputs.append({
        "name": "draw_ones_boxes",
        "images": images_7,
        "boxes": boxes_7
    })

    # Input 8: Float16 image, completely negative coords
    images_8 = np.ones((2, 15, 15, 3), dtype=np.float16)
    boxes_8 = np.array([
        [[-1.0, -1.0, -0.1, -0.1], [-2.0, -2.0, -0.5, -0.5], [-3.0, -3.0, -0.9, -0.9]],
        [[-1.0, -1.0, -0.1, -0.1], [-2.0, -2.0, -0.5, -0.5], [-3.0, -3.0, -0.9, -0.9]]
    ], dtype=np.float32)
    list_of_inputs.append({
        "name": "draw_negative_coords",
        "images": images_8,
        "boxes": boxes_8
    })

    # Input 9: Minimal image (1x1)
    images_9 = np.ones((1, 1, 1, 3), dtype=np.float32)
    boxes_9 = np.array([[[0.0, 0.0, 1.0, 1.0]]], dtype=np.float32)
    list_of_inputs.append({
        "name": "draw_minimal_image",
        "images": images_9,
        "boxes": boxes_9
    })

    # Input 10: Large batch and multiple boxes
    images_10 = np.random.rand(5, 10, 20, 3).astype(np.float32)
    boxes_10 = np.random.rand(5, 4, 4).astype(np.float32)
    list_of_inputs.append({
        "name": "draw_large_batch",
        "images": images_10,
        "boxes": boxes_10
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.DrawBoundingBoxes"] = draw_bounding_boxes_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DynamicPartition_inputs():
    list_of_inputs = []

    # Input 1: Scalar partition, 1D data
    list_of_inputs.append({
        "name": "scalar_partition",
        "data": np.array([10, 20], dtype=np.float32),
        "partitions": np.array(1, dtype=np.int32),
        "num_partitions": 2
    })

    # Input 2: Vector partition, 1D integer data
    list_of_inputs.append({
        "name": "vector_partition_int",
        "data": np.array([10, 20, 30, 40, 50], dtype=np.int32),
        "partitions": np.array([0, 0, 1, 1, 0], dtype=np.int32),
        "num_partitions": 2
    })

    # Input 3: Vector partition, multi-dimensional float data
    list_of_inputs.append({
        "name": "vector_partition_multi_dim_float",
        "data": np.random.randn(3, 4, 5).astype(np.float32),
        "partitions": np.array([0, 2, 1], dtype=np.int32),
        "num_partitions": 3
    })

    # Input 4: 2D partition, 3D float data
    list_of_inputs.append({
        "name": "2d_partition_float",
        "data": np.random.randn(2, 2, 3).astype(np.float32),
        "partitions": np.array([[0, 1], [2, 0]], dtype=np.int32),
        "num_partitions": 3
    })

    # Input 5: Large number of partitions, float64 data
    list_of_inputs.append({
        "name": "large_partitions",
        "data": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64),
        "partitions": np.array([5, 4, 3, 2, 1, 0], dtype=np.int32),
        "num_partitions": 6
    })

    # Input 6: Unused partition indices
    list_of_inputs.append({
        "name": "unused_partitions",
        "data": np.array([1, 2, 3], dtype=np.int32),
        "partitions": np.array([0, 0, 0], dtype=np.int32),
        "num_partitions": 3
    })

    # Input 7: Boolean data type
    list_of_inputs.append({
        "name": "bool_data",
        "data": np.array([True, False, True, True], dtype=np.bool_),
        "partitions": np.array([1, 0, 1, 0], dtype=np.int32),
        "num_partitions": 2
    })

    # Input 8: Complex64 data type
    list_of_inputs.append({
        "name": "complex_data",
        "data": np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64),
        "partitions": np.array([0, 1, 0], dtype=np.int32),
        "num_partitions": 2
    })

    # Input 9: 3D partition with 4D data
    partitions_3d = np.zeros((2, 2, 2), dtype=np.int32)
    partitions_3d[0, 1, 1] = 1
    partitions_3d[1, 0, 1] = 1
    list_of_inputs.append({
        "name": "3d_partition_4d_data",
        "data": np.random.randint(0, 100, size=(2, 2, 2, 5), dtype=np.int32),
        "partitions": partitions_3d,
        "num_partitions": 2
    })

    # Input 10: Single partition
    list_of_inputs.append({
        "name": "single_partition",
        "data": np.array([10, 20, 30, 40], dtype=np.int32),
        "partitions": np.array([0, 0, 0, 0], dtype=np.int32),
        "num_partitions": 1
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.DynamicPartition"] = tf_raw_ops_DynamicPartition_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Empty_inputs():
    list_of_inputs = []
    
    # Input 1
    list_of_inputs.append({
        'shape': np.array([2, 3], dtype=np.int32),
        'dtype': np.float32,
        'init': True,
        'name': "empty_1"
    })
    
    # Input 2
    list_of_inputs.append({
        'shape': np.array([5], dtype=np.int32),
        'dtype': np.int32,
        'init': False,
        'name': "empty_2"
    })
    
    # Input 3
    list_of_inputs.append({
        'shape': np.array([0, 4], dtype=np.int32),
        'dtype': np.float64,
        'init': True,
        'name': "empty_3"
    })
    
    # Input 4
    list_of_inputs.append({
        'shape': np.array([1, 1, 1], dtype=np.int32),
        'dtype': np.bool_,
        'init': False,
        'name': "empty_4"
    })
    
    # Input 5
    list_of_inputs.append({
        'shape': np.array([2, 2, 2, 2], dtype=np.int32),
        'dtype': np.int64,
        'init': True,
        'name': "empty_5"
    })
    
    # Input 6
    list_of_inputs.append({
        'shape': np.array([3, 2, 4], dtype=np.int32),
        'dtype': np.uint8,
        'init': False,
        'name': "empty_6"
    })
    
    # Input 7
    list_of_inputs.append({
        'shape': np.array([], dtype=np.int32),
        'dtype': np.float32,
        'init': True,
        'name': "empty_7"
    })
    
    # Input 8
    list_of_inputs.append({
        'shape': np.array([2, 0], dtype=np.int32),
        'dtype': np.complex64,
        'init': False,
        'name': "empty_8"
    })
    
    # Input 9
    list_of_inputs.append({
        'shape': np.array([4, 2], dtype=np.int32),
        'dtype': np.int16,
        'init': True,
        'name': "empty_9"
    })
    
    # Input 10
    list_of_inputs.append({
        'shape': np.array([100], dtype=np.int32),
        'dtype': np.float16,
        'init': False,
        'name': "empty_10"
    })
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Empty"] = tf_raw_ops_Empty_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_EnsureShape_inputs():
    list_of_inputs = []
    
    # Input 1: Scalar int32
    list_of_inputs.append({
        'name': 'ensure_shape_scalar',
        'input': np.array(42, dtype=np.int32),
        'shape': []
    })
    
    # Input 2: 1D float32
    list_of_inputs.append({
        'name': 'ensure_shape_1d',
        'input': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'shape': [3]
    })
    
    # Input 3: 2D int64
    list_of_inputs.append({
        'name': 'ensure_shape_2d',
        'input': np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64),
        'shape': [3, 2]
    })
    
    # Input 4: 3D float64
    list_of_inputs.append({
        'name': 'ensure_shape_3d',
        'input': np.zeros((2, 2, 2), dtype=np.float64),
        'shape': [2, 2, 2]
    })
    
    # Input 5: 4D int32
    list_of_inputs.append({
        'name': 'ensure_shape_4d',
        'input': np.ones((1, 3, 4, 5), dtype=np.int32),
        'shape': [1, 3, 4, 5]
    })
    
    # Input 6: 1D bool
    list_of_inputs.append({
        'name': 'ensure_shape_bool',
        'input': np.array([True, False, True], dtype=np.bool_),
        'shape': [3]
    })
    
    # Input 7: 2D float32 with negative values
    list_of_inputs.append({
        'name': 'ensure_shape_neg',
        'input': np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32),
        'shape': [2, 2]
    })
    
    # Input 8: Empty 1D tensor
    list_of_inputs.append({
        'name': 'ensure_shape_empty_1d',
        'input': np.array([], dtype=np.float32),
        'shape': [0]
    })

    # Input 9: Empty 2D tensor
    list_of_inputs.append({
        'name': 'ensure_shape_empty_2d',
        'input': np.empty((2, 0), dtype=np.int32),
        'shape': [2, 0]
    })

    # Input 10: 5D float32
    list_of_inputs.append({
        'name': 'ensure_shape_5d',
        'input': np.ones((1, 2, 1, 2, 1), dtype=np.float32),
        'shape': [1, 2, 1, 2, 1]
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.EnsureShape"] = tf_raw_ops_EnsureShape_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_FractionalMaxPool_inputs():
    list_of_inputs = []

    # Input 1
    value = np.arange(36, dtype=np.float32).reshape((1, 6, 6, 1))
    pooling_ratio = [1.0, 1.5, 1.5, 1.0]
    input_dict = {
        'pseudo_random': False,
        'overlapping': False,
        'deterministic': False,
        'seed': 0,
        'seed2': 0,
        'name': "pool1",
        'value': value,
        'pooling_ratio': pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    value = np.random.uniform(-10, 10, (1, 10, 10, 3)).astype(np.float64)
    pooling_ratio = [1.0, 2.0, 2.0, 1.0]
    input_dict = {
        'pseudo_random': True,
        'overlapping': True,
        'deterministic': True,
        'seed': 42,
        'seed2': 24,
        'name': "pool2",
        'value': value,
        'pooling_ratio': pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    value = np.random.randint(0, 100, (2, 8, 8, 2), dtype=np.int32)
    pooling_ratio = [1.0, 1.2, 1.2, 1.0]
    input_dict = {
        'pseudo_random': True,
        'overlapping': False,
        'deterministic': False,
        'seed': 0,
        'seed2': 0,
        'name': "pool3",
        'value': value,
        'pooling_ratio': pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    value = np.random.randint(-50, 50, (1, 12, 12, 1), dtype=np.int64)
    pooling_ratio = [1.0, 1.44, 1.73, 1.0]
    input_dict = {
        'pseudo_random': False,
        'overlapping': True,
        'deterministic': True,
        'seed': 1,
        'seed2': 2,
        'name': "pool4",
        'value': value,
        'pooling_ratio': pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    value = np.random.normal(0, 1, (2, 5, 5, 1)).astype(np.float32)
    pooling_ratio = [1.0, 1.1, 1.1, 1.0]
    input_dict = {
        'pseudo_random': True,
        'overlapping': True,
        'deterministic': False,
        'seed': 0,
        'seed2': 0,
        'name': "pool5",
        'value': value,
        'pooling_ratio': pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    value = np.ones((1, 15, 15, 4), dtype=np.float32) * 5.5
    pooling_ratio = [1.0, 1.8, 1.3, 1.0]
    input_dict = {
        'pseudo_random': False,
        'overlapping': False,
        'deterministic': True,
        'seed': 7,
        'seed2': 14,
        'name': "pool6",
        'value': value,
        'pooling_ratio': pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    value = np.random.randint(-10, 10, (3, 6, 6, 3), dtype=np.int32)
    pooling_ratio = [1.0, 1.5, 1.2, 1.0]
    input_dict = {
        'pseudo_random': True,
        'overlapping': False,
        'deterministic': True,
        'seed': 99,
        'seed2': 99,
        'name': "pool7",
        'value': value,
        'pooling_ratio': pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    value = np.random.uniform(-1, 1, (1, 4, 4, 1)).astype(np.float64)
    pooling_ratio = [1.0, 1.0, 1.0, 1.0]
    input_dict = {
        'pseudo_random': False,
        'overlapping': True,
        'deterministic': False,
        'seed': 0,
        'seed2': 0,
        'name': "pool8",
        'value': value,
        'pooling_ratio': pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    value = np.random.randint(0, 10, (4, 10, 10, 2), dtype=np.int64)
    pooling_ratio = [1.0, 2.5, 2.5, 1.0]
    input_dict = {
        'pseudo_random': False,
        'overlapping': False,
        'deterministic': False,
        'seed': 0,
        'seed2': 0,
        'name': "pool9",
        'value': value,
        'pooling_ratio': pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    value = np.random.normal(5, 2, (2, 20, 20, 3)).astype(np.float32)
    pooling_ratio = [1.0, 3.0, 1.5, 1.0]
    input_dict = {
        'pseudo_random': True,
        'overlapping': True,
        'deterministic': True,
        'seed': 12345,
        'seed2': 54321,
        'name': "pool10",
        'value': value,
        'pooling_ratio': pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.FractionalMaxPool"] = tf_raw_ops_FractionalMaxPool_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_InvertPermutation_inputs():
    list_of_inputs = []
    
    # Input 1
    input_dict = {
        'name': 'invert_1',
        'x': np.array([0], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input_dict = {
        'name': 'invert_2',
        'x': np.array([1, 0], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'name': 'invert_3',
        'x': np.array([2, 0, 1], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'name': 'invert_4',
        'x': np.array([3, 4, 0, 2, 1], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'name': 'invert_5',
        'x': np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'name': 'invert_6',
        'x': np.array([5, 4, 3, 2, 1, 0], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'name': 'invert_7',
        'x': np.array([1, 3, 0, 2], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'name': 'invert_8',
        'x': np.array([2, 1, 0, 4, 3], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'name': 'invert_9',
        'x': np.array([4, 3, 2, 1, 0, 5, 6], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'name': 'invert_10',
        'x': np.array([0, 2, 1, 3], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.InvertPermutation"] = tf_raw_ops_InvertPermutation_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_matrix_set_diag_inputs():
    list_of_inputs = []

    # Case 1: 2D input, float32, square matrix
    input_1 = np.zeros((3, 3), dtype=np.float32)
    diagonal_1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({
        'name': 'float32_square_2d',
        'input': input_1,
        'diagonal': diagonal_1
    })

    # Case 2: 2D input, int32, rectangular M > N
    input_2 = np.ones((4, 3), dtype=np.int32)
    diagonal_2 = np.array([-1, -2, -3], dtype=np.int32)
    list_of_inputs.append({
        'name': 'int32_rect_m_greater_n',
        'input': input_2,
        'diagonal': diagonal_2
    })

    # Case 3: 2D input, float64, rectangular M < N with negative values
    input_3 = np.array([[-1.0, -2.0, -3.0, -4.0], [-5.0, -6.0, -7.0, -8.0]], dtype=np.float64)
    diagonal_3 = np.array([10.0, 20.0], dtype=np.float64)
    list_of_inputs.append({
        'name': 'float64_rect_m_less_n',
        'input': input_3,
        'diagonal': diagonal_3
    })

    # Case 4: 3D input, float32, batch of square matrices
    input_4 = np.random.randn(2, 3, 3).astype(np.float32)
    diagonal_4 = np.random.randn(2, 3).astype(np.float32)
    list_of_inputs.append({
        'name': 'float32_batch_3d',
        'input': input_4,
        'diagonal': diagonal_4
    })

    # Case 5: 3D input, int64, batch of rectangular matrices
    input_5 = np.zeros((2, 2, 4), dtype=np.int64)
    diagonal_5 = np.ones((2, 2), dtype=np.int64)
    list_of_inputs.append({
        'name': 'int64_batch_3d',
        'input': input_5,
        'diagonal': diagonal_5
    })

    # Case 6: 4D input, complex64
    input_6 = np.ones((2, 2, 3, 3), dtype=np.complex64)
    diagonal_6 = np.zeros((2, 2, 3), dtype=np.complex64)
    list_of_inputs.append({
        'name': 'complex64_4d',
        'input': input_6,
        'diagonal': diagonal_6
    })

    # Case 7: 2D input, bool
    input_7 = np.array([[True, False], [False, True]], dtype=bool)
    diagonal_7 = np.array([False, False], dtype=bool)
    list_of_inputs.append({
        'name': 'bool_2d',
        'input': input_7,
        'diagonal': diagonal_7
    })

    # Case 8: 3D input, float32, larger dimensions
    input_8 = np.full((1, 5, 5), -1.0, dtype=np.float32)
    diagonal_8 = np.array([[10.0, 20.0, 30.0, 40.0, 50.0]], dtype=np.float32)
    list_of_inputs.append({
        'name': 'float32_large_3d',
        'input': input_8,
        'diagonal': diagonal_8
    })

    # Case 9: 2D input, uint8, small rectangular
    input_9 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.uint8)
    diagonal_9 = np.array([10, 20], dtype=np.uint8)
    list_of_inputs.append({
        'name': 'uint8_rect_2d',
        'input': input_9,
        'diagonal': diagonal_9
    })

    # Case 10: 4D input, int32, 1x1 batch size
    input_10 = np.ones((1, 1, 2, 2), dtype=np.int32)
    diagonal_10 = np.array([[[9, 9]]], dtype=np.int32)
    list_of_inputs.append({
        'name': 'int32_1x1_batch_4d',
        'input': input_10,
        'diagonal': diagonal_10
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.MatrixSetDiag"] = tf_matrix_set_diag_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MaxPool3D_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool_1',
        'input': np.random.randn(1, 2, 2, 2, 1).astype(np.float32),
        'ksize': [1, 2, 2, 2, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool_2',
        'input': np.random.randn(2, 3, 3, 3, 2).astype(np.float32),
        'ksize': [1, 2, 2, 2, 1],
        'strides': [1, 2, 2, 2, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool_3',
        'input': np.random.randn(1, 4, 4, 4, 3).astype(np.float32),
        'ksize': [1, 3, 3, 3, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool_4',
        'input': np.random.randn(1, 2, 3, 4, 1).astype(np.float32),
        'ksize': [1, 2, 2, 2, 1],
        'strides': [1, 1, 2, 2, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool_5',
        'input': np.random.randn(2, 2, 2, 2, 2).astype(np.float32),
        'ksize': [1, 1, 1, 1, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool_6',
        'input': np.random.randn(3, 5, 5, 5, 2).astype(np.float32),
        'ksize': [1, 2, 2, 2, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool_7',
        'input': np.random.randn(1, 3, 3, 3, 4).astype(np.float32),
        'ksize': [1, 2, 3, 2, 1],
        'strides': [1, 2, 1, 2, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool_8',
        'input': np.random.randn(2, 4, 4, 4, 1).astype(np.float32),
        'ksize': [1, 4, 4, 4, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool_9',
        'input': np.random.randn(1, 1, 1, 1, 1).astype(np.float32),
        'ksize': [1, 1, 1, 1, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'data_format': 'NDHWC',
        'name': 'pool_10',
        'input': np.random.randn(2, 3, 4, 5, 2).astype(np.float32),
        'ksize': [1, 1, 2, 3, 1],
        'strides': [1, 1, 2, 2, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.MaxPool3D"] = tf_raw_ops_MaxPool3D_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_MaxPoolWithArgmax_inputs():
    list_of_inputs = []
    
    # Input 1
    input_dict = {
        'Targmax': np.int64,
        'include_batch_in_index': False,
        'name': "maxpool_1",
        'input': np.random.randn(1, 4, 4, 1).astype(np.float32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': "VALID"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input_dict = {
        'Targmax': np.int32,
        'include_batch_in_index': True,
        'name': "maxpool_2",
        'input': np.random.randint(-10, 10, size=(2, 3, 3, 2)).astype(np.int32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 1, 1, 1],
        'padding': "SAME"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input_dict = {
        'Targmax': np.int64,
        'include_batch_in_index': True,
        'name': "maxpool_3",
        'input': np.random.randn(1, 5, 5, 3).astype(np.float64),
        'ksize': [1, 3, 3, 1],
        'strides': [1, 2, 2, 1],
        'padding': "VALID"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_dict = {
        'Targmax': np.int32,
        'include_batch_in_index': False,
        'name': "maxpool_4",
        'input': np.random.randint(-50, 50, size=(3, 8, 8, 4)).astype(np.int32),
        'ksize': [1, 4, 4, 1],
        'strides': [1, 4, 4, 1],
        'padding': "SAME"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_dict = {
        'Targmax': np.int64,
        'include_batch_in_index': False,
        'name': "maxpool_5",
        'input': np.random.randint(0, 255, size=(1, 2, 2, 1)).astype(np.int64),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 1, 1, 1],
        'padding': "VALID"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_dict = {
        'Targmax': np.int64,
        'include_batch_in_index': True,
        'name': "maxpool_6",
        'input': np.random.randn(2, 6, 6, 2).astype(np.float32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': "SAME"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_dict = {
        'Targmax': np.int32,
        'include_batch_in_index': False,
        'name': "maxpool_7",
        'input': np.random.randint(-5, 5, size=(1, 4, 4, 2)).astype(np.int32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': "VALID"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_dict = {
        'Targmax': np.int64,
        'include_batch_in_index': True,
        'name': "maxpool_8",
        'input': np.random.randint(0, 1000, size=(4, 4, 4, 1)).astype(np.int64),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': "SAME"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_dict = {
        'Targmax': np.int64,
        'include_batch_in_index': False,
        'name': "maxpool_9",
        'input': np.random.randn(1, 10, 10, 3).astype(np.float32),
        'ksize': [1, 5, 5, 1],
        'strides': [1, 3, 3, 1],
        'padding': "VALID"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_dict = {
        'Targmax': np.int32,
        'include_batch_in_index': True,
        'name': "maxpool_10",
        'input': np.random.randn(2, 2, 2, 2).astype(np.float64),
        'ksize': [1, 1, 1, 1],
        'strides': [1, 1, 1, 1],
        'padding': "VALID"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.MaxPoolWithArgmax"] = tf_raw_ops_MaxPoolWithArgmax_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Pad_inputs():
    list_of_inputs = []
    
    # Input 1: 1D input, int32, padding 1D with int32 paddings
    list_of_inputs.append({
        "name": "pad_1d_int32",
        "input": np.array([1, 2, 3], dtype=np.int32),
        "paddings": np.array([[1, 2]], dtype=np.int32)
    })

    # Input 2: 2D input, float32, padding 2D with int32 paddings
    list_of_inputs.append({
        "name": "pad_2d_float32",
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "paddings": np.array([[1, 1], [2, 2]], dtype=np.int32)
    })

    # Input 3: 3D input, float64, padding 3D with int64 paddings
    list_of_inputs.append({
        "name": "pad_3d_float64",
        "input": np.random.randn(2, 3, 4).astype(np.float64),
        "paddings": np.array([[0, 1], [1, 1], [2, 0]], dtype=np.int64)
    })

    # Input 4: 4D input, int64, padding 4D with int32 paddings
    list_of_inputs.append({
        "name": "pad_4d_int64",
        "input": np.random.randint(0, 10, size=(1, 2, 2, 1)).astype(np.int64),
        "paddings": np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32)
    })

    # Input 5: 1D input with zero padding
    list_of_inputs.append({
        "name": "pad_zero_padding",
        "input": np.array([5, 6, 7], dtype=np.float32),
        "paddings": np.array([[0, 0]], dtype=np.int32)
    })

    # Input 6: 2D input with large padding, int32 paddings
    list_of_inputs.append({
        "name": "pad_large_padding",
        "input": np.array([[1]], dtype=np.int32),
        "paddings": np.array([[5, 5], [5, 5]], dtype=np.int32)
    })

    # Input 7: 5D input, float32, int32 paddings
    list_of_inputs.append({
        "name": "pad_5d_float32",
        "input": np.ones((1, 1, 1, 1, 1), dtype=np.float32),
        "paddings": np.array([[1, 1], [0, 0], [2, 2], [0, 0], [1, 1]], dtype=np.int32)
    })

    # Input 8: 2D input, bool type, int32 paddings
    list_of_inputs.append({
        "name": "pad_bool",
        "input": np.array([[True, False], [False, True]], dtype=np.bool_),
        "paddings": np.array([[1, 0], [0, 1]], dtype=np.int32)
    })

    # Input 9: 3D input, complex64 type, int32 paddings
    list_of_inputs.append({
        "name": "pad_complex64",
        "input": (np.random.randn(2, 2, 2) + 1j * np.random.randn(2, 2, 2)).astype(np.complex64),
        "paddings": np.array([[0, 0], [1, 1], [0, 0]], dtype=np.int32)
    })

    # Input 10: 1D input, float16 type, int32 paddings
    list_of_inputs.append({
        "name": "pad_float16",
        "input": np.array([1.5, 2.5], dtype=np.float16),
        "paddings": np.array([[3, 3]], dtype=np.int32)
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.Pad"] = tf_raw_ops_Pad_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_PlaceholderWithDefault_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "name": "placeholder_1",
        "input": np.array([1, 2, 3], dtype=np.int32),
        "shape": [3]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "name": "placeholder_2",
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "shape": [2, 2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "name": "placeholder_3",
        "input": np.array(42, dtype=np.int64),
        "shape": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "name": "placeholder_4",
        "input": np.array([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]], dtype=np.float64),
        "shape": [2, 2, 2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "name": "placeholder_5",
        "input": np.array([True, False, True], dtype=np.bool_),
        "shape": [3]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "name": "placeholder_6",
        "input": np.array([[-1, -2], [-3, -4]], dtype=np.int16),
        "shape": [2, 2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "name": "placeholder_7",
        "input": np.ones((1, 2, 2, 1), dtype=np.uint8),
        "shape": [1, 2, 2, 1]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "name": "placeholder_8",
        "input": np.array([0.1, -0.2, 0.5], dtype=np.float16),
        "shape": [3]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "name": "placeholder_9",
        "input": np.zeros((2, 2, 2), dtype=np.int32),
        "shape": [2, 2, 2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "name": "placeholder_10",
        "input": np.ones((1, 1, 1, 1, 1), dtype=np.float32),
        "shape": [1, 1, 1, 1, 1]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.PlaceholderWithDefault"] = tf_raw_ops_PlaceholderWithDefault_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_randomuniform_inputs():
    list_of_inputs = []

    # Input 1
    list_of_inputs.append({
        'seed': 1,
        'seed2': 1,
        'name': "random_1",
        'shape': np.array([3, 3], dtype=np.int32),
        'dtype': np.float32
    })

    # Input 2
    list_of_inputs.append({
        'seed': 2,
        'seed2': 2,
        'name': "random_2",
        'shape': np.array([10], dtype=np.int64),
        'dtype': np.float64
    })

    # Input 3
    list_of_inputs.append({
        'seed': 3,
        'seed2': 3,
        'name': "random_3",
        'shape': np.array([2, 5, 2], dtype=np.int32),
        'dtype': np.float16
    })

    # Input 4
    list_of_inputs.append({
        'seed': 4,
        'seed2': 4,
        'name': "random_4",
        'shape': np.array([1, 1, 1], dtype=np.int64),
        'dtype': np.float32
    })

    # Input 5
    list_of_inputs.append({
        'seed': 5,
        'seed2': 5,
        'name': "random_5",
        'shape': np.array([8, 8], dtype=np.int32),
        'dtype': np.float64
    })

    # Input 6
    list_of_inputs.append({
        'seed': 6,
        'seed2': 6,
        'name': "random_6",
        'shape': np.array([100, 10], dtype=np.int32),
        'dtype': np.float32
    })

    # Input 7
    list_of_inputs.append({
        'seed': 7,
        'seed2': 7,
        'name': "random_7",
        'shape': np.array([1], dtype=np.int64),
        'dtype': np.float16
    })

    # Input 8
    list_of_inputs.append({
        'seed': 8,
        'seed2': 8,
        'name': "random_8",
        'shape': np.array([4, 4, 4, 4], dtype=np.int32),
        'dtype': np.float32
    })

    # Input 9
    list_of_inputs.append({
        'seed': 9,
        'seed2': 9,
        'name': "random_9",
        'shape': np.array([2, 3], dtype=np.int64),
        'dtype': np.float64
    })

    # Input 10
    list_of_inputs.append({
        'seed': 10,
        'seed2': 10,
        'name': "random_10",
        'shape': np.array([1000], dtype=np.int32),
        'dtype': np.float32
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.RandomUniform"] = tf_raw_ops_randomuniform_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ResizeArea_inputs():
    list_of_inputs = []

    # Input 1: Float32 input, downsampling
    images = np.random.rand(1, 10, 10, 3).astype(np.float32)
    size = np.array([5, 5], dtype=np.int32)
    align_corners = False
    name = "resize_1"
    list_of_inputs.append({
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "name": name
    })

    # Input 2: UInt8 input, downsampling with align_corners=True
    images = np.random.randint(0, 256, size=(2, 8, 8, 1)).astype(np.uint8)
    size = np.array([4, 4], dtype=np.int32)
    align_corners = True
    name = "resize_2"
    list_of_inputs.append({
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "name": name
    })

    # Input 3: Int32 input with negative values
    images = np.random.randint(-100, 100, size=(1, 16, 16, 4)).astype(np.int32)
    size = np.array([8, 8], dtype=np.int32)
    align_corners = False
    name = "resize_3"
    list_of_inputs.append({
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "name": name
    })

    # Input 4: Float64 input
    images = np.random.rand(3, 12, 12, 3).astype(np.float64)
    size = np.array([6, 6], dtype=np.int32)
    align_corners = True
    name = "resize_4"
    list_of_inputs.append({
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "name": name
    })

    # Input 5: Float32 input, other dimensions
    images = np.random.rand(1, 20, 20, 2).astype(np.float32)
    size = np.array([10, 10], dtype=np.int32)
    align_corners = False
    name = "resize_5"
    list_of_inputs.append({
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "name": name
    })

    # Input 6: UInt8 input with multiple channels
    images = np.random.randint(0, 256, size=(2, 6, 6, 3)).astype(np.uint8)
    size = np.array([3, 3], dtype=np.int32)
    align_corners = True
    name = "resize_6"
    list_of_inputs.append({
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "name": name
    })

    # Input 7: Int32 input
    images = np.random.randint(-50, 50, size=(1, 4, 4, 1)).astype(np.int32)
    size = np.array([2, 2], dtype=np.int32)
    align_corners = False
    name = "resize_7"
    list_of_inputs.append({
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "name": name
    })

    # Input 8: Float64 input with larger batch
    images = np.random.rand(4, 14, 14, 3).astype(np.float64)
    size = np.array([7, 7], dtype=np.int32)
    align_corners = True
    name = "resize_8"
    list_of_inputs.append({
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "name": name
    })

    # Input 9: Float32 input, scaling to same size
    images = np.random.rand(1, 22, 22, 3).astype(np.float32)
    size = np.array([22, 22], dtype=np.int32)
    align_corners = False
    name = "resize_9"
    list_of_inputs.append({
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "name": name
    })

    # Input 10: Float32 input, upsampling
    images = np.random.rand(2, 2, 2, 2).astype(np.float32)
    size = np.array([4, 4], dtype=np.int32)
    align_corners = True
    name = "resize_10"
    list_of_inputs.append({
        "images": images,
        "size": size,
        "align_corners": align_corners,
        "name": name
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.ResizeArea"] = tf_raw_ops_ResizeArea_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ResizeBilinear_inputs():
    list_of_inputs = []
    
    # Input 1
    input_dict = {
        "images": np.ones((1, 2, 2, 1), dtype=np.float32),
        "size": np.array([4, 4], dtype=np.int32),
        "align_corners": False,
        "half_pixel_centers": False,
        "name": "resize_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input_dict = {
        "images": np.zeros((2, 3, 3, 3), dtype=np.uint8),
        "size": np.array([6, 6], dtype=np.int32),
        "align_corners": True,
        "half_pixel_centers": False,
        "name": "resize_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input_dict = {
        "images": np.arange(48, dtype=np.uint8).reshape((1, 4, 4, 3)),
        "size": np.array([2, 2], dtype=np.int32),
        "align_corners": False,
        "half_pixel_centers": True,
        "name": "resize_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_dict = {
        "images": np.ones((2, 5, 5, 2), dtype=np.float32) * -1.5,
        "size": np.array([10, 10], dtype=np.int32),
        "align_corners": False,
        "half_pixel_centers": True,
        "name": "resize_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_dict = {
        "images": np.ones((1, 1, 1, 1), dtype=np.int32),
        "size": np.array([3, 3], dtype=np.int32),
        "align_corners": False,
        "half_pixel_centers": False,
        "name": "resize_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_dict = {
        "images": np.arange(16, dtype=np.float32).reshape((1, 2, 2, 4)),
        "size": np.array([5, 5], dtype=np.int32),
        "align_corners": True,
        "half_pixel_centers": False,
        "name": "resize_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_dict = {
        "images": np.zeros((1, 8, 8, 1), dtype=np.int32),
        "size": np.array([4, 4], dtype=np.int32),
        "align_corners": False,
        "half_pixel_centers": True,
        "name": "resize_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_dict = {
        "images": np.ones((2, 2, 2, 3), dtype=np.float32) * 3.14,
        "size": np.array([4, 4], dtype=np.int32),
        "align_corners": True,
        "half_pixel_centers": False,
        "name": "resize_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_dict = {
        "images": np.ones((1, 3, 3, 2), dtype=np.uint8),
        "size": np.array([1, 1], dtype=np.int32),
        "align_corners": False,
        "half_pixel_centers": False,
        "name": "resize_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_dict = {
        "images": np.arange(32, dtype=np.float32).reshape((1, 4, 4, 2)),
        "size": np.array([8, 8], dtype=np.int32),
        "align_corners": False,
        "half_pixel_centers": True,
        "name": "resize_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.ResizeBilinear"] = tf_raw_ops_ResizeBilinear_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ResizeNearestNeighbor_inputs():
    list_of_inputs = []

    # Input 1
    images = np.random.rand(1, 2, 2, 3).astype(np.float32)
    size = np.array([4, 4], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False
    name = "resize_1"
    list_of_inputs.append({
        'align_corners': align_corners,
        'half_pixel_centers': half_pixel_centers,
        'name': name,
        'images': images,
        'size': size
    })

    # Input 2
    images = np.random.randint(0, 255, size=(2, 4, 4, 1)).astype(np.int32)
    size = np.array([2, 2], dtype=np.int32)
    align_corners = True
    half_pixel_centers = False
    name = "resize_2"
    list_of_inputs.append({
        'align_corners': align_corners,
        'half_pixel_centers': half_pixel_centers,
        'name': name,
        'images': images,
        'size': size
    })

    # Input 3
    images = np.random.rand(1, 10, 10, 3).astype(np.float64)
    size = np.array([5, 5], dtype=np.int32)
    align_corners = False
    half_pixel_centers = True
    name = "resize_3"
    list_of_inputs.append({
        'align_corners': align_corners,
        'half_pixel_centers': half_pixel_centers,
        'name': name,
        'images': images,
        'size': size
    })

    # Input 4
    images = np.random.randint(-128, 127, size=(4, 8, 8, 4)).astype(np.int32)
    size = np.array([16, 16], dtype=np.int32)
    align_corners = False
    half_pixel_centers = True
    name = "resize_4"
    list_of_inputs.append({
        'align_corners': align_corners,
        'half_pixel_centers': half_pixel_centers,
        'name': name,
        'images': images,
        'size': size
    })

    # Input 5
    images = np.random.rand(1, 3, 3, 2).astype(np.float32)
    size = np.array([6, 6], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False
    name = "resize_5"
    list_of_inputs.append({
        'align_corners': align_corners,
        'half_pixel_centers': half_pixel_centers,
        'name': name,
        'images': images,
        'size': size
    })

    # Input 6
    images = np.random.randint(-1000, 1000, size=(2, 5, 5, 3)).astype(np.int32)
    size = np.array([10, 10], dtype=np.int32)
    align_corners = True
    half_pixel_centers = False
    name = "resize_6"
    list_of_inputs.append({
        'align_corners': align_corners,
        'half_pixel_centers': half_pixel_centers,
        'name': name,
        'images': images,
        'size': size
    })

    # Input 7
    images = np.random.randint(-1000, 1000, size=(1, 2, 2, 1)).astype(np.int32)
    size = np.array([1, 1], dtype=np.int32)
    align_corners = False
    half_pixel_centers = True
    name = "resize_7"
    list_of_inputs.append({
        'align_corners': align_corners,
        'half_pixel_centers': half_pixel_centers,
        'name': name,
        'images': images,
        'size': size
    })

    # Input 8
    images = np.random.rand(1, 4, 4, 3).astype(np.float64)
    size = np.array([8, 8], dtype=np.int32)
    align_corners = False
    half_pixel_centers = True
    name = "resize_8"
    list_of_inputs.append({
        'align_corners': align_corners,
        'half_pixel_centers': half_pixel_centers,
        'name': name,
        'images': images,
        'size': size
    })

    # Input 9
    images = np.random.rand(1, 3, 3, 3).astype(np.float32)
    size = np.array([10, 10], dtype=np.int32)
    align_corners = False
    half_pixel_centers = False
    name = "resize_9"
    list_of_inputs.append({
        'align_corners': align_corners,
        'half_pixel_centers': half_pixel_centers,
        'name': name,
        'images': images,
        'size': size
    })

    # Input 10
    images = np.random.rand(2, 6, 6, 1).astype(np.float32)
    size = np.array([3, 3], dtype=np.int32)
    align_corners = True
    half_pixel_centers = False
    name = "resize_10"
    list_of_inputs.append({
        'align_corners': align_corners,
        'half_pixel_centers': half_pixel_centers,
        'name': name,
        'images': images,
        'size': size
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.ResizeNearestNeighbor"] = tf_raw_ops_ResizeNearestNeighbor_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SampleDistortedBoundingBox_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "seed": 12,
        "seed2": 34,
        "min_object_covered": 0.1,
        "aspect_ratio_range": [0.75, 1.33],
        "area_range": [0.05, 1.0],
        "max_attempts": 100,
        "use_image_if_no_bounding_boxes": False,
        "name": "sample_1",
        "image_size": np.array([256, 256, 3], dtype=np.int32),
        "bounding_boxes": np.array([[[0.1, 0.1, 0.9, 0.9]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "seed": 42,
        "seed2": 42,
        "min_object_covered": 0.5,
        "aspect_ratio_range": [0.5, 2.0],
        "area_range": [0.1, 0.9],
        "max_attempts": 50,
        "use_image_if_no_bounding_boxes": True,
        "name": "sample_2",
        "image_size": np.array([512, 512, 1], dtype=np.int64),
        "bounding_boxes": np.array([[[0.0, 0.0, 1.0, 1.0]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "seed": 1,
        "seed2": 2,
        "min_object_covered": 0.25,
        "aspect_ratio_range": [0.8, 1.25],
        "area_range": [0.2, 0.8],
        "max_attempts": 200,
        "use_image_if_no_bounding_boxes": False,
        "name": "sample_3",
        "image_size": np.array([100, 100, 4], dtype=np.int16),
        "bounding_boxes": np.array([[[0.2, 0.2, 0.8, 0.8], [0.1, 0.1, 0.5, 0.5]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "seed": 123,
        "seed2": 456,
        "min_object_covered": 0.0,
        "aspect_ratio_range": [0.9, 1.1],
        "area_range": [0.3, 0.7],
        "max_attempts": 10,
        "use_image_if_no_bounding_boxes": True,
        "name": "sample_4",
        "image_size": np.array([64, 64, 3], dtype=np.int32),
        "bounding_boxes": np.array([[[0.2, 0.2, 0.8, 0.8]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "seed": 7,
        "seed2": 8,
        "min_object_covered": 0.3,
        "aspect_ratio_range": [0.6, 1.6],
        "area_range": [0.15, 0.85],
        "max_attempts": 150,
        "use_image_if_no_bounding_boxes": False,
        "name": "sample_5",
        "image_size": np.array([32, 32, 3], dtype=np.uint8),
        "bounding_boxes": np.array([[[0.0, 0.0, 0.5, 0.5]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "seed": 10,
        "seed2": 20,
        "min_object_covered": 0.0,
        "aspect_ratio_range": [0.5, 1.5],
        "area_range": [0.4, 0.9],
        "max_attempts": 20,
        "use_image_if_no_bounding_boxes": True,
        "name": "sample_6",
        "image_size": np.array([120, 120, 3], dtype=np.int8),
        "bounding_boxes": np.array([[[0.15, 0.15, 0.85, 0.85]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "seed": 100,
        "seed2": 200,
        "min_object_covered": 0.75,
        "aspect_ratio_range": [0.3, 3.0],
        "area_range": [0.01, 1.0],
        "max_attempts": 500,
        "use_image_if_no_bounding_boxes": False,
        "name": "sample_7",
        "image_size": np.array([480, 640, 3], dtype=np.int32),
        "bounding_boxes": np.array([[[0.05, 0.05, 0.95, 0.95]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "seed": 99,
        "seed2": 99,
        "min_object_covered": 0.9,
        "aspect_ratio_range": [1.0, 1.0],
        "area_range": [0.05, 0.1],
        "max_attempts": 300,
        "use_image_if_no_bounding_boxes": True,
        "name": "sample_8",
        "image_size": np.array([1000, 1000, 3], dtype=np.int32),
        "bounding_boxes": np.array([[[0.25, 0.25, 0.75, 0.75]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "seed": 1234,
        "seed2": 5678,
        "min_object_covered": 0.05,
        "aspect_ratio_range": [0.1, 10.0],
        "area_range": [0.05, 1.0],
        "max_attempts": 1000,
        "use_image_if_no_bounding_boxes": False,
        "name": "sample_9",
        "image_size": np.array([16, 16, 3], dtype=np.int32),
        "bounding_boxes": np.array([[[0.1, 0.1, 0.9, 0.9]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "seed": 111,
        "seed2": 222,
        "min_object_covered": 0.4,
        "aspect_ratio_range": [0.7, 1.4],
        "area_range": [0.1, 0.9],
        "max_attempts": 80,
        "use_image_if_no_bounding_boxes": False,
        "name": "sample_10",
        "image_size": np.array([1920, 1080, 3], dtype=np.int32),
        "bounding_boxes": np.array([[[0.3, 0.3, 0.7, 0.7]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.SampleDistortedBoundingBox"] = tf_raw_ops_SampleDistortedBoundingBox_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SegmentMin_inputs():
    list_of_inputs = []
    
    # Input 1: 1D float32 data, int32 segment_ids
    list_of_inputs.append({
        'name': 'seg_min_1',
        'data': np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        'segment_ids': np.array([0, 0, 1, 1, 2], dtype=np.int32)
    })
    
    # Input 2: 2D float32 data, int32 segment_ids
    list_of_inputs.append({
        'name': 'seg_min_2',
        'data': np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        'segment_ids': np.array([0, 0, 1], dtype=np.int32)
    })
    
    # Input 3: 3D float32 data, int64 segment_ids
    list_of_inputs.append({
        'name': 'seg_min_3',
        'data': np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32),
        'segment_ids': np.array([0, 1], dtype=np.int64)
    })
    
    # Input 4: 1D float64 data, int32 segment_ids
    list_of_inputs.append({
        'name': 'seg_min_4',
        'data': np.array([-1.5, -2.5, -3.5], dtype=np.float64),
        'segment_ids': np.array([0, 0, 1], dtype=np.int32)
    })
    
    # Input 5: 2D float64 data, int64 segment_ids
    list_of_inputs.append({
        'name': 'seg_min_5',
        'data': np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64),
        'segment_ids': np.array([0, 1], dtype=np.int64)
    })
    
    # Input 6: 3D float64 data, int32 segment_ids
    list_of_inputs.append({
        'name': 'seg_min_6',
        'data': np.ones((2, 2, 2), dtype=np.float64),
        'segment_ids': np.array([0, 1], dtype=np.int32)
    })
    
    # Input 7: 1D int32 data, int32 segment_ids
    list_of_inputs.append({
        'name': 'seg_min_7',
        'data': np.array([10, 20, 30], dtype=np.int32),
        'segment_ids': np.array([0, 1, 1], dtype=np.int32)
    })
    
    # Input 8: 2D int32 data, int64 segment_ids
    list_of_inputs.append({
        'name': 'seg_min_8',
        'data': np.array([[5, 6], [1, 2], [7, 8]], dtype=np.int32),
        'segment_ids': np.array([0, 0, 1], dtype=np.int64)
    })
    
    # Input 9: 1D int64 data, int64 segment_ids
    list_of_inputs.append({
        'name': 'seg_min_9',
        'data': np.array([-10, -20, -30], dtype=np.int64),
        'segment_ids': np.array([0, 1, 1], dtype=np.int64)
    })
    
    # Input 10: 2D int64 data, int32 segment_ids
    list_of_inputs.append({
        'name': 'seg_min_10',
        'data': np.array([[100, 200], [300, 400]], dtype=np.int64),
        'segment_ids': np.array([0, 1], dtype=np.int32)
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.SegmentMin"] = tf_raw_ops_SegmentMin_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseSoftmaxCrossEntropyWithLogits_inputs():
    list_of_inputs = []

    # Input 1
    features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    labels = np.array([0, 2], dtype=np.int32)
    name = "loss_1"
    list_of_inputs.append({"name": name, "features": features, "labels": labels})

    # Input 2
    features = np.array([[-1.0, 0.0], [2.5, -3.2], [0.1, 0.2], [10.0, 11.0]], dtype=np.float64)
    labels = np.array([1, 0, 1, 0], dtype=np.int64)
    name = "loss_2"
    list_of_inputs.append({"name": name, "features": features, "labels": labels})

    # Input 3
    features = np.arange(10, dtype=np.float16).reshape((1, 10))
    labels = np.array([5], dtype=np.int32)
    name = "loss_3"
    list_of_inputs.append({"name": name, "features": features, "labels": labels})

    # Input 4
    features = np.random.uniform(-5.0, 5.0, (10, 5)).astype(np.float32)
    labels = np.random.randint(0, 5, size=(10,), dtype=np.int64)
    name = "loss_4"
    list_of_inputs.append({"name": name, "features": features, "labels": labels})

    # Input 5
    features = np.zeros((3, 3), dtype=np.float64)
    labels = np.array([0, 1, 2], dtype=np.int32)
    name = "loss_5"
    list_of_inputs.append({"name": name, "features": features, "labels": labels})

    # Input 6
    features = np.array([[100.0, -100.0, 0.0, 1.0], 
                         [-50.0, 50.0, 2.0, -2.0], 
                         [0.5, -0.5, 0.1, -0.1], 
                         [10.0, 20.0, 30.0, 40.0], 
                         [-1.0, -2.0, -3.0, -4.0]], dtype=np.float32)
    labels = np.array([2, 1, 3, 0, 2], dtype=np.int32)
    name = "loss_6"
    list_of_inputs.append({"name": name, "features": features, "labels": labels})

    # Input 7
    features = np.eye(8, dtype=np.float16)
    labels = np.arange(8, dtype=np.int64)
    name = "loss_7"
    list_of_inputs.append({"name": name, "features": features, "labels": labels})

    # Input 8
    features = np.random.normal(0.0, 1.0, (15, 2)).astype(np.float32)
    labels = np.random.randint(0, 2, size=(15,), dtype=np.int32)
    name = "loss_8"
    list_of_inputs.append({"name": name, "features": features, "labels": labels})

    # Input 9
    features = np.ones((2, 20), dtype=np.float64)
    labels = np.array([19, 0], dtype=np.int64)
    name = "loss_9"
    list_of_inputs.append({"name": name, "features": features, "labels": labels})

    # Input 10
    features = np.array([[1.5], [-2.3], [0.0], [9.1], [-0.5], [4.4]], dtype=np.float32)
    labels = np.zeros((6,), dtype=np.int32)
    name = "loss_10"
    list_of_inputs.append({"name": name, "features": features, "labels": labels})

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits"] = tf_raw_ops_SparseSoftmaxCrossEntropyWithLogits_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Sum_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float32 array, sum along axis 0
    input_val = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axis_val = np.array([0], dtype=np.int32)
    keep_dims = False
    name = "sum_1"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis_val,
        "keep_dims": keep_dims,
        "name": name
    })

    # Input 2: 2D int32 array with negative values, keep dimensions
    input_val = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int32)
    axis_val = np.array([1], dtype=np.int32)
    keep_dims = True
    name = "sum_2"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis_val,
        "keep_dims": keep_dims,
        "name": name
    })

    # Input 3: 3D float64 array, reduce multiple axes
    input_val = np.array([[[1.5, 2.5], [3.5, 4.5]], [[5.5, 6.5], [7.5, 8.5]]], dtype=np.float64)
    axis_val = np.array([0, 2], dtype=np.int64)
    keep_dims = False
    name = "sum_3"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis_val,
        "keep_dims": keep_dims,
        "name": name
    })

    # Input 4: 1D int64 array, scalar reduction axis
    input_val = np.array([-10, 20, -30, 40], dtype=np.int64)
    axis_val = np.array([0], dtype=np.int32)
    keep_dims = True
    name = "sum_4"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis_val,
        "keep_dims": keep_dims,
        "name": name
    })

    # Input 5: Complex numbers (complex64) sum
    input_val = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    axis_val = np.array([1], dtype=np.int32)
    keep_dims = False
    name = "sum_5"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis_val,
        "keep_dims": keep_dims,
        "name": name
    })

    # Input 6: Unsigned integers (uint8), sum along axis 0
    input_val = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    axis_val = np.array([0], dtype=np.int64)
    keep_dims = True
    name = "sum_6"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis_val,
        "keep_dims": keep_dims,
        "name": name
    })

    # Input 7: 3D int16 array, negative indexing for axis
    input_val = np.array([[[1, -2], [3, -4]], [[5, -6], [7, -8]]], dtype=np.int16)
    axis_val = np.array([-1], dtype=np.int32)
    keep_dims = False
    name = "sum_7"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis_val,
        "keep_dims": keep_dims,
        "name": name
    })

    # Input 8: High dimensional random float32 array, fully reduced (all axes)
    input_val = np.random.randn(2, 3, 4).astype(np.float32)
    axis_val = np.array([0, 1, 2], dtype=np.int32)
    keep_dims = False
    name = "sum_8"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis_val,
        "keep_dims": keep_dims,
        "name": name
    })

    # Input 9: Small int8 array with negative axis
    input_val = np.array([-5, 5, -10, 10], dtype=np.int8)
    axis_val = np.array([-1], dtype=np.int64)
    keep_dims = True
    name = "sum_9"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis_val,
        "keep_dims": keep_dims,
        "name": name
    })

    # Input 10: 4D array of ones, complex reduction
    input_val = np.ones((2, 2, 2, 2), dtype=np.float32)
    axis_val = np.array([1, 3], dtype=np.int32)
    keep_dims = True
    name = "sum_10"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis_val,
        "keep_dims": keep_dims,
        "name": name
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.Sum"] = tf_raw_ops_Sum_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_TensorSummary_inputs():
    list_of_inputs = []

    # Input 1
    list_of_inputs.append({
        'tensor': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'description': "",
        'labels': [],
        'display_name': "",
        'name': "summary_1"
    })

    # Input 2
    list_of_inputs.append({
        'tensor': np.array([[1, 2], [3, 4]], dtype=np.int32),
        'description': "json-encoded-proto",
        'labels': [],
        'display_name': "my_tensor",
        'name': "summary_2"
    })

    # Input 3
    list_of_inputs.append({
        'tensor': np.array(42.0, dtype=np.float64),
        'description': "",
        'labels': [],
        'display_name': "scalar",
        'name': "summary_3"
    })

    # Input 4
    list_of_inputs.append({
        'tensor': np.array([True, False, True], dtype=np.bool_),
        'description': "{}",
        'labels': [],
        'display_name': "boolean_tensor",
        'name': "summary_4"
    })

    # Input 5
    list_of_inputs.append({
        'tensor': np.array([10, 20, 30], dtype=np.int16),
        'description': "",
        'labels': [],
        'display_name': "int16_tensor",
        'name': "summary_5"
    })

    # Input 6
    list_of_inputs.append({
        'tensor': np.random.randn(2, 3, 4).astype(np.float32),
        'description': "desc",
        'labels': [],
        'display_name': "3d_tensor",
        'name': "summary_6"
    })

    # Input 7
    list_of_inputs.append({
        'tensor': np.array([-1, -2, -3], dtype=np.int64),
        'description': "",
        'labels': [],
        'display_name': "negatives",
        'name': "summary_7"
    })

    # Input 8
    list_of_inputs.append({
        'tensor': np.array([1e-5, 1e-6], dtype=np.float16),
        'description': "{\"key\": \"val\"}",
        'labels': [],
        'display_name': "half_precision",
        'name': "summary_8"
    })

    # Input 9
    list_of_inputs.append({
        'tensor': np.zeros((1, 1, 1, 1), dtype=np.int32),
        'description': "",
        'labels': [],
        'display_name': "four_d",
        'name': "summary_9"
    })

    # Input 10
    list_of_inputs.append({
        'tensor': np.array([255, 128, 0], dtype=np.uint8),
        'description': "",
        'labels': [],
        'display_name': "image_bytes",
        'name': "summary_10"
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.TensorSummary"] = tf_raw_ops_TensorSummary_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_TruncatedNormal_inputs():
    list_of_inputs = []

    # Input 1
    list_of_inputs.append({
        'shape': np.array([2, 3], dtype=np.int32),
        'dtype': np.float32,
        'seed': 1,
        'seed2': 2,
        'name': "truncated_normal_1"
    })

    # Input 2
    list_of_inputs.append({
        'shape': np.array([5], dtype=np.int64),
        'dtype': np.float64,
        'seed': 3,
        'seed2': 4,
        'name': "truncated_normal_2"
    })

    # Input 3
    list_of_inputs.append({
        'shape': np.array([3, 3, 3], dtype=np.int32),
        'dtype': np.float16,
        'seed': 42,
        'seed2': 123,
        'name': "truncated_normal_3"
    })

    # Input 4
    list_of_inputs.append({
        'shape': np.array([1], dtype=np.int32),
        'dtype': np.float32,
        'seed': 10,
        'seed2': 20,
        'name': "truncated_normal_4"
    })

    # Input 5
    list_of_inputs.append({
        'shape': np.array([10, 10], dtype=np.int64),
        'dtype': np.float32,
        'seed': 100,
        'seed2': 200,
        'name': "truncated_normal_5"
    })

    # Input 6
    list_of_inputs.append({
        'shape': np.array([1, 2, 3, 4], dtype=np.int32),
        'dtype': np.float64,
        'seed': 7,
        'seed2': 14,
        'name': "truncated_normal_6"
    })

    # Input 7
    list_of_inputs.append({
        'shape': np.array([], dtype=np.int32),
        'dtype': np.float32,
        'seed': 5,
        'seed2': 1,
        'name': "truncated_normal_7"
    })

    # Input 8
    list_of_inputs.append({
        'shape': np.array([100], dtype=np.int32),
        'dtype': np.float16,
        'seed': 999,
        'seed2': 888,
        'name': "truncated_normal_8"
    })

    # Input 9
    list_of_inputs.append({
        'shape': np.array([2, 2, 2, 2, 2], dtype=np.int64),
        'dtype': np.float32,
        'seed': 12345,
        'seed2': 54321,
        'name': "truncated_normal_9"
    })

    # Input 10
    list_of_inputs.append({
        'shape': np.array([4, 1], dtype=np.int32),
        'dtype': np.float64,
        'seed': 12,
        'seed2': 34,
        'name': "truncated_normal_10"
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.TruncatedNormal"] = tf_raw_ops_TruncatedNormal_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_reverse_inputs():
    list_of_inputs = []

    # Input 1: 1-D float32, axis=[0]
    list_of_inputs.append({
        "tensor": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "axis": np.array([0], dtype=np.int32),
        "name": "reverse_1d_float32"
    })

    # Input 2: 1-D int32, axis=[-1]
    list_of_inputs.append({
        "tensor": np.array([10, 20, 30], dtype=np.int32),
        "axis": np.array([-1], dtype=np.int32),
        "name": "reverse_1d_int32"
    })

    # Input 3: 2-D bool, axis=[0]
    list_of_inputs.append({
        "tensor": np.array([[True, False], [False, True]], dtype=np.bool_),
        "axis": np.array([0], dtype=np.int32),
        "name": "reverse_2d_bool"
    })

    # Input 4: 2-D uint8, axis=[1]
    list_of_inputs.append({
        "tensor": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
        "axis": np.array([1], dtype=np.int32),
        "name": "reverse_2d_uint8"
    })

    # Input 5: 2-D float64, axis=[0, 1]
    list_of_inputs.append({
        "tensor": np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64),
        "axis": np.array([0, 1], dtype=np.int32),
        "name": "reverse_2d_multi"
    })

    # Input 6: 3-D int64, axis=[0, 2]
    list_of_inputs.append({
        "tensor": np.arange(8, dtype=np.int64).reshape((2, 2, 2)),
        "axis": np.array([0, 2], dtype=np.int64),
        "name": "reverse_3d_int64"
    })

    # Input 7: 3-D float32, axis=[-1, -3]
    list_of_inputs.append({
        "tensor": np.arange(24, dtype=np.float32).reshape((2, 3, 4)),
        "axis": np.array([-1, -3], dtype=np.int32),
        "name": "reverse_3d_neg"
    })

    # Input 8: 4-D complex64, axis=[1, 3]
    list_of_inputs.append({
        "tensor": np.arange(16, dtype=np.complex64).reshape((2, 2, 2, 2)),
        "axis": np.array([1, 3], dtype=np.int32),
        "name": "reverse_4d_complex"
    })

    # Input 9: 5-D string tensor, axis=[4]
    list_of_inputs.append({
        "tensor": np.array([[[[["a", "b"], ["c", "d"]]]]], dtype=np.object_),
        "axis": np.array([4], dtype=np.int32),
        "name": "reverse_5d_string"
    })

    # Input 10: 8-D int16, axis=[7]
    list_of_inputs.append({
        "tensor": np.arange(256, dtype=np.int16).reshape((2, 2, 2, 2, 2, 2, 2, 2)),
        "axis": np.array([7], dtype=np.int32),
        "name": "reverse_8d_max"
    })

    # Input 11: 2-D with empty axis
    list_of_inputs.append({
        "tensor": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "axis": np.array([], dtype=np.int32),
        "name": "reverse_empty_axis"
    })

    return list_of_inputs

generated_inputs["tf.reverse"] = tf_reverse_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_hann_window_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'window_length': np.array(8, dtype=np.int32),
        'periodic': True,
        'dtype': tf.float32,
        'name': "hann_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'window_length': np.array(16, dtype=np.int32),
        'periodic': False,
        'dtype': tf.float64,
        'name': "hann_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'window_length': np.array(1, dtype=np.int32),
        'periodic': True,
        'dtype': tf.float16,
        'name': "hann_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'window_length': np.array(64, dtype=np.int32),
        'periodic': False,
        'dtype': tf.float32,
        'name': "hann_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'window_length': np.array(128, dtype=np.int32),
        'periodic': True,
        'dtype': tf.float64,
        'name': "hann_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'window_length': np.array(3, dtype=np.int32),
        'periodic': False,
        'dtype': tf.float16,
        'name': "hann_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'window_length': np.array(1024, dtype=np.int32),
        'periodic': True,
        'dtype': tf.float32,
        'name': "hann_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'window_length': np.array(256, dtype=np.int32),
        'periodic': False,
        'dtype': tf.float32,
        'name': "hann_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'window_length': np.array(5, dtype=np.int32),
        'periodic': True,
        'dtype': tf.float64,
        'name': "hann_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'window_length': np.array(512, dtype=np.int32),
        'periodic': False,
        'dtype': tf.float64,
        'name': "hann_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.signal.hann_window"] = tf_signal_hann_window_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_irfft2d_inputs():
    list_of_inputs = []

    # Input 1, valid 2D complex64
    input_tensor = (np.random.randn(8, 5) + 1j * np.random.randn(8, 5)).astype(np.complex64)
    fft_length = np.array([8, 8], dtype=np.int32)
    name = "irfft2d_1"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 2, valid 2D complex128, odd length
    input_tensor = (np.random.randn(6, 3) + 1j * np.random.randn(6, 3)).astype(np.complex128)
    fft_length = np.array([6, 5], dtype=np.int32)
    name = "irfft2d_2"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 3, valid 3D complex64
    input_tensor = (np.random.randn(2, 4, 3) + 1j * np.random.randn(2, 4, 3)).astype(np.complex64)
    fft_length = np.array([4, 4], dtype=np.int32)
    name = "irfft2d_3"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 4, valid 3D complex128, negative values
    input_tensor = (np.random.randn(3, 10, 6) + 1j * np.random.randn(3, 10, 6)).astype(np.complex128)
    fft_length = np.array([10, 11], dtype=np.int32)
    name = "irfft2d_4"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 5, valid 4D complex64
    input_tensor = (np.random.randn(1, 2, 8, 5) + 1j * np.random.randn(1, 2, 8, 5)).astype(np.complex64)
    fft_length = np.array([8, 8], dtype=np.int32)
    name = "irfft2d_5"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 6, cropping case
    input_tensor = (np.random.randn(6, 5) + 1j * np.random.randn(6, 5)).astype(np.complex64)
    fft_length = np.array([4, 4], dtype=np.int32)
    name = "irfft2d_6"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 7, padding case
    input_tensor = (np.random.randn(4, 3) + 1j * np.random.randn(4, 3)).astype(np.complex128)
    fft_length = np.array([8, 8], dtype=np.int32)
    name = "irfft2d_7"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 8, odd dimensions
    input_tensor = (np.random.randn(7, 4) + 1j * np.random.randn(7, 4)).astype(np.complex64)
    fft_length = np.array([7, 7], dtype=np.int32)
    name = "irfft2d_8"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 9, larger size
    input_tensor = (np.random.randn(32, 17) + 1j * np.random.randn(32, 17)).astype(np.complex64)
    fft_length = np.array([32, 32], dtype=np.int32)
    name = "irfft2d_9"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 10, multibatch with mixed dimensions
    input_tensor = (np.random.randn(2, 2, 5, 4) + 1j * np.random.randn(2, 2, 5, 4)).astype(np.complex128)
    fft_length = np.array([5, 6], dtype=np.int32)
    name = "irfft2d_10"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    return list_of_inputs

generated_inputs["tf.signal.irfft2d"] = tf_signal_irfft2d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_irfft3d_inputs():
    list_of_inputs = []
    
    # Input 1
    input_tensor = np.ones((4, 4, 3), dtype=np.complex64)
    fft_length = np.array([4, 4, 4], dtype=np.int32)
    name = "irfft3d_1"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 2
    input_tensor = np.ones((3, 3, 2), dtype=np.complex128)
    fft_length = np.array([3, 3, 3], dtype=np.int32)
    name = "irfft3d_2"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 3
    input_tensor = (np.random.randn(2, 5, 5).astype(np.complex64) + 
                    1j * np.random.randn(2, 5, 5).astype(np.complex64))
    fft_length = np.array([2, 5, 8], dtype=np.int32)
    name = "irfft3d_3"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 4
    input_tensor = (np.random.randn(2, 8, 8, 5).astype(np.complex128) + 
                    1j * np.random.randn(2, 8, 8, 5).astype(np.complex128))
    fft_length = np.array([8, 8, 8], dtype=np.int32)
    name = "irfft3d_4"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 5
    input_tensor = np.ones((1, 1, 6, 6, 4), dtype=np.complex64)
    fft_length = np.array([6, 6, 6], dtype=np.int32)
    name = "irfft3d_5"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 6
    input_tensor = np.zeros((2, 2, 2), dtype=np.complex64)
    fft_length = np.array([2, 2, 2], dtype=np.int32)
    name = "irfft3d_6"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 7
    input_tensor = np.ones((5, 5, 10), dtype=np.complex128)
    fft_length = np.array([5, 5, 18], dtype=np.int32)
    name = "irfft3d_7"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 8
    input_tensor = np.ones((4, 4, 4, 3), dtype=np.complex64)
    fft_length = np.array([4, 4, 5], dtype=np.int32)
    name = "irfft3d_8"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 9
    input_tensor = np.ones((10, 10, 6), dtype=np.complex128)
    fft_length = np.array([10, 10, 10], dtype=np.int32)
    name = "irfft3d_9"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 10
    input_tensor = np.ones((8, 12, 7), dtype=np.complex64)
    fft_length = np.array([8, 12, 12], dtype=np.int32)
    name = "irfft3d_10"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    return list_of_inputs

generated_inputs["tf.signal.irfft3d"] = tf_signal_irfft3d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_rfft2d_inputs():
    list_of_inputs = []

    # Input 1: Basic float32, 2D input, exact match fft_length
    input_tensor = np.random.randn(4, 4).astype(np.float32)
    fft_length = np.array([4, 4], dtype=np.int32)
    name = "rfft2d_1"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 2: float32, negative values, 2D input
    input_tensor = np.array([[-1.5, -2.5], [-3.5, -4.5]], dtype=np.float32)
    fft_length = np.array([2, 2], dtype=np.int32)
    name = "rfft2d_2"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 3: float64, 3D input, exact match fft_length
    input_tensor = np.random.randn(2, 3, 3).astype(np.float64)
    fft_length = np.array([3, 3], dtype=np.int32)
    name = "rfft2d_3"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 4: float32, 2D input, fft_length is larger (zero-padding)
    input_tensor = np.random.randn(3, 3).astype(np.float32)
    fft_length = np.array([5, 5], dtype=np.int32)
    name = "rfft2d_4"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 5: float32, 2D input, fft_length is smaller (cropping)
    input_tensor = np.random.randn(6, 6).astype(np.float32)
    fft_length = np.array([4, 4], dtype=np.int32)
    name = "rfft2d_5"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 6: float64, 4D input, exact match fft_length
    input_tensor = np.random.randn(2, 2, 4, 4).astype(np.float64)
    fft_length = np.array([4, 4], dtype=np.int32)
    name = "rfft2d_6"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 7: float32, 2D input, one dimension cropped, one dimension padded
    input_tensor = np.random.randn(4, 8).astype(np.float32)
    fft_length = np.array([6, 6], dtype=np.int32)
    name = "rfft2d_7"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 8: Small float32 input
    input_tensor = np.random.randn(2, 2).astype(np.float32)
    fft_length = np.array([2, 2], dtype=np.int32)
    name = "rfft2d_8"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 9: float64, 3D input, mixed cropping and padding
    input_tensor = np.random.randn(1, 10, 10).astype(np.float64)
    fft_length = np.array([5, 12], dtype=np.int32)
    name = "rfft2d_9"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 10: float32, 5D input
    input_tensor = np.random.randn(2, 2, 2, 3, 3).astype(np.float32)
    fft_length = np.array([3, 3], dtype=np.int32)
    name = "rfft2d_10"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    return list_of_inputs

generated_inputs["tf.signal.rfft2d"] = tf_signal_rfft2d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_rfft3d_inputs():
    list_of_inputs = []

    # Input 1: Basic 3D float32 input, exact fft_length
    input_tensor = np.random.randn(4, 4, 4).astype(np.float32)
    fft_length = np.array([4, 4, 4], dtype=np.int32)
    name = "rfft3d_1"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 2: 4D float32 input (batch dimension), exact fft_length
    input_tensor = np.random.randn(2, 8, 8, 8).astype(np.float32)
    fft_length = np.array([8, 8, 8], dtype=np.int32)
    name = "rfft3d_2"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 3: float64 input, exact fft_length
    input_tensor = np.random.randn(3, 5, 7).astype(np.float64)
    fft_length = np.array([3, 5, 7], dtype=np.int32)
    name = "rfft3d_3"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 4: High dimensional float32 input
    input_tensor = np.random.randn(1, 2, 2, 4, 4, 4).astype(np.float32)
    fft_length = np.array([4, 4, 4], dtype=np.int32)
    name = "rfft3d_4"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 5: Crop along all dimensions (fft_length smaller than input)
    input_tensor = np.random.randn(8, 8, 8).astype(np.float32)
    fft_length = np.array([4, 4, 4], dtype=np.int32)
    name = "rfft3d_5"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 6: Pad along all dimensions (fft_length larger than input)
    input_tensor = np.random.randn(4, 4, 4).astype(np.float32)
    fft_length = np.array([8, 8, 8], dtype=np.int32)
    name = "rfft3d_6"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 7: Mixed crop and pad, float64
    input_tensor = np.random.randn(2, 4, 8, 16).astype(np.float64)
    fft_length = np.array([6, 4, 20], dtype=np.int32)
    name = "rfft3d_7"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 8: Another valid size mix, float32
    input_tensor = np.random.randn(10, 10, 10).astype(np.float32)
    fft_length = np.array([5, 12, 8], dtype=np.int32)
    name = "rfft3d_8"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 9: Large powers of 2 (efficient for FFT)
    input_tensor = np.random.randn(16, 16, 16).astype(np.float32)
    fft_length = np.array([16, 16, 16], dtype=np.int32)
    name = "rfft3d_9"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    # Input 10: 4D float64 input, exact fft_length
    input_tensor = np.random.randn(2, 2, 2, 2).astype(np.float64)
    fft_length = np.array([2, 2, 2], dtype=np.int32)
    name = "rfft3d_10"
    list_of_inputs.append({
        "input_tensor": input_tensor,
        "fft_length": fft_length,
        "name": name
    })

    return list_of_inputs

generated_inputs["tf.signal.rfft3d"] = tf_signal_rfft3d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_tensordot_inputs():
    list_of_inputs = []
    
    # Input 1
    input_dict = {
        "a": np.random.randn(3, 4).astype(np.float32),
        "b": np.random.randn(4, 5).astype(np.float32),
        "axes": [[1], [0]],
        "name": "tensordot_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input_dict = {
        "a": np.random.randn(3, 4, 5).astype(np.float32),
        "b": np.random.randn(4, 3, 2).astype(np.float32),
        "axes": [[1, 0], [0, 1]],
        "name": "tensordot_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input_dict = {
        "a": np.random.randn(5, 5).astype(np.float64),
        "b": np.random.randn(5, 5).astype(np.float64),
        "axes": [[0], [0]],
        "name": "tensordot_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_dict = {
        "a": np.random.randn(2, 3, 4).astype(np.float32),
        "b": np.random.randn(3, 4, 5).astype(np.float32),
        "axes": [[1, 2], [0, 1]],
        "name": "tensordot_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_dict = {
        "a": np.random.randn(10).astype(np.float32),
        "b": np.random.randn(10).astype(np.float32),
        "axes": [[0], [0]],
        "name": "tensordot_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_dict = {
        "a": np.random.randn(2, 2, 2).astype(np.float64),
        "b": np.random.randn(2, 2, 2).astype(np.float64),
        "axes": [[0, 1, 2], [0, 1, 2]],
        "name": "tensordot_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_dict = {
        "a": np.random.randn(4, 2).astype(np.float32),
        "b": np.random.randn(2, 3).astype(np.float32),
        "axes": [[1], [0]],
        "name": "tensordot_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_dict = {
        "a": np.random.randn(1, 5, 1).astype(np.float64),
        "b": np.random.randn(5, 1, 1).astype(np.float64),
        "axes": [[1], [0]],
        "name": "tensordot_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_dict = {
        "a": np.random.randn(2, 3).astype(np.float32),
        "b": np.random.randn(3, 4).astype(np.float32),
        "axes": [[1], [0]],
        "name": "tensordot_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_dict = {
        "a": np.random.randn(3, 3, 3).astype(np.float32),
        "b": np.random.randn(3, 3, 3).astype(np.float32),
        "axes": [[2], [1]],
        "name": "tensordot_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.tensordot_1"] = tf_tensordot_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_image_extract_patches_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float32, VALID padding, 3x3 size, stride 1, rate 1
    images = np.arange(1 * 10 * 10 * 1, dtype=np.float32).reshape((1, 10, 10, 1))
    sizes = [1, 3, 3, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = 'VALID'
    name = "patch_extract_1"
    list_of_inputs.append({
        'images': images,
        'sizes': sizes,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 2: SAME padding, int32 dtype, stride 2
    images = np.arange(1 * 8 * 8 * 1, dtype=np.int32).reshape((1, 8, 8, 1))
    sizes = [1, 3, 3, 1]
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = 'SAME'
    name = "patch_extract_2"
    list_of_inputs.append({
        'images': images,
        'sizes': sizes,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 3: Multiple channels (RGB), float32, larger size, valid padding
    images = np.random.rand(1, 16, 16, 3).astype(np.float32)
    sizes = [1, 5, 5, 1]
    strides = [1, 5, 5, 1]
    rates = [1, 1, 1, 1]
    padding = 'VALID'
    name = "patch_extract_3"
    list_of_inputs.append({
        'images': images,
        'sizes': sizes,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 4: Dilated patches (rate > 1), float32
    images = np.arange(1 * 12 * 12 * 1, dtype=np.float32).reshape((1, 12, 12, 1))
    sizes = [1, 3, 3, 1]
    strides = [1, 2, 2, 1]
    rates = [1, 2, 2, 1]
    padding = 'VALID'
    name = "patch_extract_4"
    list_of_inputs.append({
        'images': images,
        'sizes': sizes,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 5: Batch size > 1, float32, SAME padding
    images = np.random.rand(2, 6, 6, 2).astype(np.float32)
    sizes = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = 'SAME'
    name = "patch_extract_5"
    list_of_inputs.append({
        'images': images,
        'sizes': sizes,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 6: Large strides, int32 dtype, VALID padding
    images = np.arange(1 * 15 * 15 * 1, dtype=np.int32).reshape((1, 15, 15, 1))
    sizes = [1, 4, 4, 1]
    strides = [1, 6, 6, 1]
    rates = [1, 1, 1, 1]
    padding = 'VALID'
    name = "patch_extract_6"
    list_of_inputs.append({
        'images': images,
        'sizes': sizes,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 7: Small image matching size, float32, SAME padding
    images = np.random.rand(1, 4, 4, 1).astype(np.float32)
    sizes = [1, 4, 4, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = 'SAME'
    name = "patch_extract_7"
    list_of_inputs.append({
        'images': images,
        'sizes': sizes,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 8: Batch size > 1, multi-channel, int32 dtype
    images = np.random.randint(-100, 100, size=(3, 10, 10, 4), dtype=np.int32)
    sizes = [1, 3, 3, 1]
    strides = [1, 3, 3, 1]
    rates = [1, 1, 1, 1]
    padding = 'VALID'
    name = "patch_extract_8"
    list_of_inputs.append({
        'images': images,
        'sizes': sizes,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 9: High dilation rate, SAME padding, float32 dtype
    images = np.arange(1 * 20 * 20 * 1, dtype=np.float32).reshape((1, 20, 20, 1))
    sizes = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 4, 4, 1]
    padding = 'SAME'
    name = "patch_extract_9"
    list_of_inputs.append({
        'images': images,
        'sizes': sizes,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    # Input 10: 1x1 patches (effectively reshaping the image), float32
    images = np.random.randn(1, 8, 8, 3).astype(np.float32)
    sizes = [1, 1, 1, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = 'VALID'
    name = "patch_extract_10"
    list_of_inputs.append({
        'images': images,
        'sizes': sizes,
        'strides': strides,
        'rates': rates,
        'padding': padding,
        'name': name
    })

    return list_of_inputs

generated_inputs["tf.image.extract_patches"] = tf_image_extract_patches_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_nn_atrous_conv2d_transpose_inputs():
    list_of_inputs = []

    # Input 1: Basic Float32, Padding 'SAME', Rate 2
    value = np.random.randn(2, 5, 5, 3).astype(np.float32)
    filters = np.random.randn(3, 3, 4, 3).astype(np.float32)
    output_shape = np.array([2, 5, 5, 4], dtype=np.int32)
    rate = 2
    padding = 'SAME'
    name = 'atrous_conv_transpose_1'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'output_shape': output_shape,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 2: Float32, Padding 'VALID', Rate 1
    value = np.random.randn(1, 4, 4, 2).astype(np.float32)
    filters = np.random.randn(3, 3, 5, 2).astype(np.float32)
    output_shape = np.array([1, 6, 6, 5], dtype=np.int32)
    rate = 1
    padding = 'VALID'
    name = 'atrous_conv_transpose_2'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'output_shape': output_shape,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 3: Float32, Padding 'VALID', Rate 2
    value = np.random.randn(2, 3, 3, 1).astype(np.float32)
    filters = np.random.randn(2, 2, 2, 1).astype(np.float32)
    output_shape = np.array([2, 5, 5, 2], dtype=np.int32)
    rate = 2
    padding = 'VALID'
    name = 'atrous_conv_transpose_3'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'output_shape': output_shape,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 4: Large Rate, Padding 'SAME', Rate 3
    value = np.random.randn(1, 8, 8, 4).astype(np.float32)
    filters = np.random.randn(3, 3, 2, 4).astype(np.float32)
    output_shape = np.array([1, 8, 8, 2], dtype=np.int32)
    rate = 3
    padding = 'SAME'
    name = 'atrous_conv_transpose_4'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'output_shape': output_shape,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 5: Float64 type
    value = np.random.randn(3, 10, 10, 2).astype(np.float64)
    filters = np.random.randn(5, 5, 3, 2).astype(np.float64)
    output_shape = np.array([3, 10, 10, 3], dtype=np.int32)
    rate = 1
    padding = 'SAME'
    name = 'atrous_conv_transpose_5'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'output_shape': output_shape,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 6: Asymmetric Filters, Padding 'VALID', Rate 3
    value = np.random.randn(1, 2, 2, 1).astype(np.float32)
    filters = np.random.randn(2, 3, 1, 1).astype(np.float32)
    output_shape = np.array([1, 5, 8, 1], dtype=np.int32)
    rate = 3
    padding = 'VALID'
    name = 'atrous_conv_transpose_6'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'output_shape': output_shape,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 7: Negative values
    value = np.random.uniform(-10.0, -1.0, (1, 3, 3, 1)).astype(np.float32)
    filters = np.random.uniform(-5.0, -0.5, (2, 2, 1, 1)).astype(np.float32)
    output_shape = np.array([1, 3, 3, 1], dtype=np.int32)
    rate = 2
    padding = 'SAME'
    name = 'atrous_conv_transpose_7'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'output_shape': output_shape,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 8: 1x1 base resolution, high rate
    value = np.random.randn(1, 1, 1, 1).astype(np.float32)
    filters = np.random.randn(2, 2, 1, 1).astype(np.float32)
    output_shape = np.array([1, 6, 6, 1], dtype=np.int32)
    rate = 5
    padding = 'VALID'
    name = 'atrous_conv_transpose_8'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'output_shape': output_shape,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 9: Large batch and high channel sizes
    value = np.random.randn(16, 14, 14, 32).astype(np.float32)
    filters = np.random.randn(3, 3, 64, 32).astype(np.float32)
    output_shape = np.array([16, 14, 14, 64], dtype=np.int32)
    rate = 2
    padding = 'SAME'
    name = 'atrous_conv_transpose_9'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'output_shape': output_shape,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    # Input 10: 1x1 filter size, rate 2, Padding 'VALID'
    value = np.random.randn(2, 5, 5, 16).astype(np.float32)
    filters = np.random.randn(1, 1, 8, 16).astype(np.float32)
    output_shape = np.array([2, 5, 5, 8], dtype=np.int32)
    rate = 2
    padding = 'VALID'
    name = 'atrous_conv_transpose_10'
    list_of_inputs.append({
        'value': value,
        'filters': filters,
        'output_shape': output_shape,
        'rate': rate,
        'padding': padding,
        'name': name
    })

    return list_of_inputs

generated_inputs["tf.nn.atrous_conv2d_transpose"] = tf_nn_atrous_conv2d_transpose_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_nn_depthwise_conv2d_backprop_input_inputs():
    list_of_inputs = []
    
    # 1. NHWC, SAME, stride 1, float32
    input_sizes = np.array([2, 8, 8, 3], dtype=np.int32)
    filter_val = np.random.randn(3, 3, 3, 2).astype(np.float32)
    out_backprop = np.random.randn(2, 8, 8, 6).astype(np.float32)
    input_dict = {
        'input_sizes': input_sizes,
        'filter': filter_val,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2. NHWC, VALID, stride 1, float32
    input_sizes = np.array([1, 5, 5, 2], dtype=np.int32)
    filter_val = np.random.randn(3, 3, 2, 1).astype(np.float32)
    out_backprop = np.random.randn(1, 3, 3, 2).astype(np.float32)
    input_dict = {
        'input_sizes': input_sizes,
        'filter': filter_val,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3. NHWC, SAME, stride 2, float32
    input_sizes = np.array([2, 6, 6, 4], dtype=np.int32)
    filter_val = np.random.randn(3, 3, 4, 2).astype(np.float32)
    out_backprop = np.random.randn(2, 3, 3, 8).astype(np.float32)
    input_dict = {
        'input_sizes': input_sizes,
        'filter': filter_val,
        'out_backprop': out_backprop,
        'strides': [1, 2, 2, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4. NHWC, VALID, stride 1, float32, smaller size
    input_sizes = np.array([1, 4, 4, 1], dtype=np.int32)
    filter_val = np.random.randn(2, 2, 1, 1).astype(np.float32)
    out_backprop = np.random.randn(1, 3, 3, 1).astype(np.float32)
    input_dict = {
        'input_sizes': input_sizes,
        'filter': filter_val,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5. NHWC, VALID, stride 2, float32
    input_sizes = np.array([1, 6, 6, 3], dtype=np.int32)
    filter_val = np.random.randn(3, 3, 3, 1).astype(np.float32)
    out_backprop = np.random.randn(1, 2, 2, 3).astype(np.float32)
    input_dict = {
        'input_sizes': input_sizes,
        'filter': filter_val,
        'out_backprop': out_backprop,
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6. NHWC, SAME, stride 1, multiplier=2
    input_sizes = np.array([2, 4, 4, 2], dtype=np.int32)
    filter_val = np.random.randn(2, 2, 2, 2).astype(np.float32)
    out_backprop = np.random.randn(2, 4, 4, 4).astype(np.float32)
    input_dict = {
        'input_sizes': input_sizes,
        'filter': filter_val,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7. NHWC, VALID, stride 2, multiplier=3
    input_sizes = np.array([1, 7, 7, 1], dtype=np.int32)
    filter_val = np.random.randn(3, 3, 1, 3).astype(np.float32)
    out_backprop = np.random.randn(1, 3, 3, 3).astype(np.float32)
    input_dict = {
        'input_sizes': input_sizes,
        'filter': filter_val,
        'out_backprop': out_backprop,
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8. float64 type, NHWC, SAME, stride 1
    input_sizes = np.array([2, 4, 4, 1], dtype=np.int32)
    filter_val = np.random.randn(2, 2, 1, 1).astype(np.float64)
    out_backprop = np.random.randn(2, 4, 4, 1).astype(np.float64)
    input_dict = {
        'input_sizes': input_sizes,
        'filter': filter_val,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9. float16 type, NHWC, VALID, stride 1
    input_sizes = np.array([1, 3, 3, 2], dtype=np.int32)
    filter_val = np.random.randn(2, 2, 2, 2).astype(np.float16)
    out_backprop = np.random.randn(1, 2, 2, 4).astype(np.float16)
    input_dict = {
        'input_sizes': input_sizes,
        'filter': filter_val,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10. float32 with negative values, NHWC, VALID
    input_sizes = np.array([1, 4, 4, 1], dtype=np.int32)
    filter_val = np.array([[[[-1.0]], [[2.0]]], [[[0.5]], [[-1.5]]]], dtype=np.float32)
    out_backprop = np.random.randn(1, 3, 3, 1).astype(np.float32)
    input_dict = {
        'input_sizes': input_sizes,
        'filter': filter_val,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'test10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.depthwise_conv2d_backprop_input"] = tf_nn_depthwise_conv2d_backprop_input_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_separable_conv2d_inputs():
    list_of_inputs = []

    # Input 1: Basic NHWC format with same padding
    input_val = np.random.randn(2, 5, 5, 3).astype(np.float32)
    depthwise_filter = np.random.randn(3, 3, 3, 2).astype(np.float32)
    pointwise_filter = np.random.randn(1, 1, 6, 4).astype(np.float32)
    input_dict = {
        'input': input_val,
        'depthwise_filter': depthwise_filter,
        'pointwise_filter': pointwise_filter,
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'conv_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Strides greater than 1, valid padding
    input_val = np.random.randn(1, 10, 10, 2).astype(np.float32)
    depthwise_filter = np.random.randn(2, 2, 2, 1).astype(np.float32)
    pointwise_filter = np.random.randn(1, 1, 2, 3).astype(np.float32)
    input_dict = {
        'input': input_val,
        'depthwise_filter': depthwise_filter,
        'pointwise_filter': pointwise_filter,
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'conv_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Atrous depthwise convolution (dilations > 1)
    input_val = np.random.randn(1, 8, 8, 4).astype(np.float32)
    depthwise_filter = np.random.randn(3, 3, 4, 1).astype(np.float32)
    pointwise_filter = np.random.randn(1, 1, 4, 2).astype(np.float32)
    input_dict = {
        'input': input_val,
        'depthwise_filter': depthwise_filter,
        'pointwise_filter': pointwise_filter,
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [2, 2],
        'name': 'conv_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Double precision float64 type
    input_val = np.random.randn(2, 6, 6, 3).astype(np.float64)
    depthwise_filter = np.random.randn(2, 2, 3, 2).astype(np.float64)
    pointwise_filter = np.random.randn(1, 1, 6, 3).astype(np.float64)
    input_dict = {
        'input': input_val,
        'depthwise_filter': depthwise_filter,
        'pointwise_filter': pointwise_filter,
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'conv_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: NCHW format
    input_val = np.random.randn(2, 3, 5, 5).astype(np.float32)
    depthwise_filter = np.random.randn(3, 3, 3, 2).astype(np.float32)
    pointwise_filter = np.random.randn(1, 1, 6, 4).astype(np.float32)
    input_dict = {
        'input': input_val,
        'depthwise_filter': depthwise_filter,
        'pointwise_filter': pointwise_filter,
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NCHW',
        'dilations': [1, 1],
        'name': 'conv_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: NCHW format with stride 2
    input_val = np.random.randn(1, 2, 8, 8).astype(np.float32)
    depthwise_filter = np.random.randn(2, 2, 2, 1).astype(np.float32)
    pointwise_filter = np.random.randn(1, 1, 2, 2).astype(np.float32)
    input_dict = {
        'input': input_val,
        'depthwise_filter': depthwise_filter,
        'pointwise_filter': pointwise_filter,
        'strides': [1, 1, 2, 2],
        'padding': 'VALID',
        'data_format': 'NCHW',
        'dilations': [1, 1],
        'name': 'conv_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Uniformly distributed inputs with custom names
    input_val = np.random.uniform(-1, 1, (1, 7, 7, 3)).astype(np.float32)
    depthwise_filter = np.random.uniform(-1, 1, (3, 3, 3, 2)).astype(np.float32)
    pointwise_filter = np.random.uniform(-1, 1, (1, 1, 6, 5)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'depthwise_filter': depthwise_filter,
        'pointwise_filter': pointwise_filter,
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'conv_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger channel multiplier
    input_val = np.random.randn(2, 4, 4, 2).astype(np.float32)
    depthwise_filter = np.random.randn(2, 2, 2, 4).astype(np.float32)
    pointwise_filter = np.random.randn(1, 1, 8, 2).astype(np.float32)
    input_dict = {
        'input': input_val,
        'depthwise_filter': depthwise_filter,
        'pointwise_filter': pointwise_filter,
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'conv_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1x1 depthwise filter
    input_val = np.random.randn(1, 5, 5, 3).astype(np.float32)
    depthwise_filter = np.random.randn(1, 1, 3, 1).astype(np.float32)
    pointwise_filter = np.random.randn(1, 1, 3, 3).astype(np.float32)
    input_dict = {
        'input': input_val,
        'depthwise_filter': depthwise_filter,
        'pointwise_filter': pointwise_filter,
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1],
        'name': 'conv_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Dilation and NCHW format
    input_val = np.random.randn(2, 2, 6, 6).astype(np.float32)
    depthwise_filter = np.random.randn(3, 3, 2, 2).astype(np.float32)
    pointwise_filter = np.random.randn(1, 1, 4, 3).astype(np.float32)
    input_dict = {
        'input': input_val,
        'depthwise_filter': depthwise_filter,
        'pointwise_filter': pointwise_filter,
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NCHW',
        'dilations': [2, 2],
        'name': 'conv_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.separable_conv2d"] = tf_nn_separable_conv2d_inputs()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_Conv2DBackpropFilter_inputs():
    list_of_inputs = []

    # Case 1: NHWC, SAME, float32, stride 1
    input_val = np.random.randn(2, 4, 4, 3).astype(np.float32)
    filter_sizes = np.array([3, 3, 3, 8], dtype=np.int32)
    out_backprop = np.random.randn(2, 4, 4, 8).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter_sizes': filter_sizes,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': "SAME",
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv2d_bprop_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: NHWC, VALID, float32, stride 1
    input_val = np.random.randn(1, 5, 5, 2).astype(np.float32)
    filter_sizes = np.array([3, 3, 2, 4], dtype=np.int32)
    out_backprop = np.random.randn(1, 3, 3, 4).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter_sizes': filter_sizes,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'use_cudnn_on_gpu': False,
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv2d_bprop_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: NHWC, VALID, float32, stride 2
    input_val = np.random.randn(2, 6, 6, 3).astype(np.float32)
    filter_sizes = np.array([2, 2, 3, 2], dtype=np.int32)
    out_backprop = np.random.randn(2, 3, 3, 2).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter_sizes': filter_sizes,
        'out_backprop': out_backprop,
        'strides': [1, 2, 2, 1],
        'padding': "VALID",
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv2d_bprop_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: NHWC, VALID, float32, stride 1 (changed from NCHW)
    input_val = np.random.randn(2, 5, 5, 3).astype(np.float32)
    filter_sizes = np.array([3, 3, 3, 4], dtype=np.int32)
    out_backprop = np.random.randn(2, 3, 3, 4).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter_sizes': filter_sizes,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv2d_bprop_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: NHWC, SAME, float32, stride 2
    input_val = np.random.randn(1, 8, 8, 4).astype(np.float32)
    filter_sizes = np.array([4, 4, 4, 2], dtype=np.int32)
    out_backprop = np.random.randn(1, 4, 4, 2).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter_sizes': filter_sizes,
        'out_backprop': out_backprop,
        'strides': [1, 2, 2, 1],
        'padding': "SAME",
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv2d_bprop_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: NHWC, VALID, float32, stride 1, dilation 2
    input_val = np.random.randn(1, 7, 7, 2).astype(np.float32)
    filter_sizes = np.array([3, 3, 2, 2], dtype=np.int32)
    out_backprop = np.random.randn(1, 3, 3, 2).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter_sizes': filter_sizes,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 2, 2, 1],
        'name': "conv2d_bprop_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: NHWC, EXPLICIT, float32, stride 1
    input_val = np.random.randn(1, 3, 3, 1).astype(np.float32)
    filter_sizes = np.array([3, 3, 1, 1], dtype=np.int32)
    out_backprop = np.random.randn(1, 3, 3, 1).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter_sizes': filter_sizes,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': "EXPLICIT",
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [0, 0, 1, 1, 1, 1, 0, 0],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv2d_bprop_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: NHWC, EXPLICIT, float32, stride 1 (changed from NCHW)
    input_val = np.random.randn(1, 3, 3, 1).astype(np.float32)
    filter_sizes = np.array([3, 3, 1, 1], dtype=np.int32)
    out_backprop = np.random.randn(1, 3, 3, 1).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter_sizes': filter_sizes,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': "EXPLICIT",
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [0, 0, 1, 1, 1, 1, 0, 0],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv2d_bprop_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: NHWC, VALID, float64, stride 1
    input_val = np.random.randn(1, 4, 4, 1).astype(np.float64)
    filter_sizes = np.array([2, 2, 1, 1], dtype=np.int32)
    out_backprop = np.random.randn(1, 3, 3, 1).astype(np.float64)
    input_dict = {
        'input': input_val,
        'filter_sizes': filter_sizes,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv2d_bprop_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: NHWC, SAME, float16, stride 1
    input_val = np.random.randn(2, 5, 5, 3).astype(np.float16)
    filter_sizes = np.array([3, 3, 3, 4], dtype=np.int32)
    out_backprop = np.random.randn(2, 5, 5, 4).astype(np.float16)
    input_dict = {
        'input': input_val,
        'filter_sizes': filter_sizes,
        'out_backprop': out_backprop,
        'strides': [1, 1, 1, 1],
        'padding': "SAME",
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv2d_bprop_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Conv2DBackpropFilter"] = tf_raw_ops_Conv2DBackpropFilter_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_Conv2DBackpropInput_inputs():
    list_of_inputs = []
    np.random.seed(42)

    # Input 1: NHWC, VALID, float32
    input_dict = {
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_1',
        'input_sizes': np.array([2, 5, 5, 3], dtype=np.int32),
        'filter': np.random.randn(3, 3, 3, 8).astype(np.float32),
        'out_backprop': np.random.randn(2, 3, 3, 8).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: NHWC, SAME, stride 2, float32
    input_dict = {
        'strides': [1, 2, 2, 1],
        'padding': 'SAME',
        'use_cudnn_on_gpu': False,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_2',
        'input_sizes': np.array([2, 6, 6, 3], dtype=np.int32),
        'filter': np.random.randn(3, 3, 3, 8).astype(np.float32),
        'out_backprop': np.random.randn(2, 3, 3, 8).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: NHWC, VALID, float32 (originally NCHW, converted to NHWC)
    input_dict = {
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_3',
        'input_sizes': np.array([2, 5, 5, 3], dtype=np.int32),
        'filter': np.random.randn(3, 3, 3, 8).astype(np.float32),
        'out_backprop': np.random.randn(2, 3, 3, 8).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: NHWC, EXPLICIT, float32
    input_dict = {
        'strides': [1, 1, 1, 1],
        'padding': 'EXPLICIT',
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [0, 0, 1, 2, 3, 4, 0, 0],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_4',
        'input_sizes': np.array([2, 5, 5, 3], dtype=np.int32),
        'filter': np.random.randn(3, 3, 3, 8).astype(np.float32),
        'out_backprop': np.random.randn(2, 6, 10, 8).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: NHWC, dilation = 2, VALID, float32
    input_dict = {
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 2, 2, 1],
        'name': 'conv_5',
        'input_sizes': np.array([2, 5, 5, 3], dtype=np.int32),
        'filter': np.random.randn(3, 3, 3, 8).astype(np.float32),
        'out_backprop': np.random.randn(2, 1, 1, 8).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, NHWC, VALID
    input_dict = {
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'use_cudnn_on_gpu': False,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_6',
        'input_sizes': np.array([1, 4, 4, 1], dtype=np.int32),
        'filter': np.random.randn(2, 2, 1, 1).astype(np.float64),
        'out_backprop': np.random.randn(1, 3, 3, 1).astype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: NHWC, SAME, stride 2, float32 (originally NCHW, converted to NHWC)
    input_dict = {
        'strides': [1, 2, 2, 1],
        'padding': 'SAME',
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_7',
        'input_sizes': np.array([2, 8, 8, 4], dtype=np.int32),
        'filter': np.random.randn(3, 3, 4, 16).astype(np.float32),
        'out_backprop': np.random.randn(2, 4, 4, 16).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: NHWC, EXPLICIT, float32 (originally NCHW, converted to NHWC)
    input_dict = {
        'strides': [1, 1, 1, 1],
        'padding': 'EXPLICIT',
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [0, 0, 1, 1, 2, 2, 0, 0],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_8',
        'input_sizes': np.array([2, 6, 6, 2], dtype=np.int32),
        'filter': np.random.randn(3, 3, 2, 4).astype(np.float32),
        'out_backprop': np.random.randn(2, 6, 8, 4).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: NHWC, dilation with strides, float32
    input_dict = {
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'use_cudnn_on_gpu': False,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 2, 2, 1],
        'name': 'conv_9',
        'input_sizes': np.array([4, 7, 7, 3], dtype=np.int32),
        'filter': np.random.randn(3, 3, 3, 2).astype(np.float32),
        'out_backprop': np.random.randn(4, 2, 2, 2).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, larger channels, NHWC, SAME
    input_dict = {
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'use_cudnn_on_gpu': True,
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_10',
        'input_sizes': np.array([1, 14, 14, 64], dtype=np.int32),
        'filter': np.random.randn(3, 3, 64, 128).astype(np.float32),
        'out_backprop': np.random.randn(1, 14, 14, 128).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Conv2DBackpropInput"] = tf_raw_ops_Conv2DBackpropInput_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_Conv3DBackpropFilterV2_inputs():
    list_of_inputs = []

    # Input 1: NDHWC, float32, VALID padding, strides=1, dilations=1
    input_dict_1 = {
        "input": np.random.randn(2, 5, 5, 5, 3).astype(np.float32),
        "filter_sizes": np.array([3, 3, 3, 3, 4], dtype=np.int32),
        "out_backprop": np.random.randn(2, 3, 3, 3, 4).astype(np.float32),
        "strides": [1, 1, 1, 1, 1],
        "padding": "VALID",
        "data_format": "NDHWC",
        "dilations": [1, 1, 1, 1, 1],
        "name": "conv3d_backprop_filter_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: NDHWC, float32, VALID padding, strides=2, dilations=1
    input_dict_2 = {
        "input": np.random.randn(1, 8, 8, 8, 1).astype(np.float32),
        "filter_sizes": np.array([2, 2, 2, 1, 2], dtype=np.int32),
        "out_backprop": np.random.randn(1, 4, 4, 4, 2).astype(np.float32),
        "strides": [1, 2, 2, 2, 1],
        "padding": "VALID",
        "data_format": "NDHWC",
        "dilations": [1, 1, 1, 1, 1],
        "name": "conv3d_backprop_filter_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: NDHWC, float32, SAME padding, strides=1, dilations=1
    input_dict_3 = {
        "input": np.random.randn(2, 5, 5, 5, 3).astype(np.float32),
        "filter_sizes": np.array([3, 3, 3, 3, 4], dtype=np.int32),
        "out_backprop": np.random.randn(2, 5, 5, 5, 4).astype(np.float32),
        "strides": [1, 1, 1, 1, 1],
        "padding": "SAME",
        "data_format": "NDHWC",
        "dilations": [1, 1, 1, 1, 1],
        "name": "conv3d_backprop_filter_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: NDHWC, float32, SAME padding, strides=2, dilations=1
    input_dict_4 = {
        "input": np.random.randn(1, 6, 6, 6, 2).astype(np.float32),
        "filter_sizes": np.array([3, 3, 3, 2, 2], dtype=np.int32),
        "out_backprop": np.random.randn(1, 3, 3, 3, 2).astype(np.float32),
        "strides": [1, 2, 2, 2, 1],
        "padding": "SAME",
        "data_format": "NDHWC",
        "dilations": [1, 1, 1, 1, 1],
        "name": "conv3d_backprop_filter_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: NDHWC, float64, VALID padding, strides=1, dilations=1
    input_dict_5 = {
        "input": np.random.randn(2, 4, 4, 4, 2).astype(np.float64),
        "filter_sizes": np.array([2, 2, 2, 2, 1], dtype=np.int32),
        "out_backprop": np.random.randn(2, 3, 3, 3, 1).astype(np.float64),
        "strides": [1, 1, 1, 1, 1],
        "padding": "VALID",
        "data_format": "NDHWC",
        "dilations": [1, 1, 1, 1, 1],
        "name": "conv3d_backprop_filter_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: NDHWC, float16, VALID padding, strides=1, dilations=1
    input_dict_6 = {
        "input": np.random.randn(1, 3, 3, 3, 1).astype(np.float16),
        "filter_sizes": np.array([2, 2, 2, 1, 1], dtype=np.int32),
        "out_backprop": np.random.randn(1, 2, 2, 2, 1).astype(np.float16),
        "strides": [1, 1, 1, 1, 1],
        "padding": "VALID",
        "data_format": "NDHWC",
        "dilations": [1, 1, 1, 1, 1],
        "name": "conv3d_backprop_filter_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: NCDHW, float32, VALID padding, strides=1, dilations=1
    input_dict_7 = {
        "input": np.random.randn(2, 3, 5, 5, 5).astype(np.float32),
        "filter_sizes": np.array([3, 3, 3, 3, 4], dtype=np.int32),
        "out_backprop": np.random.randn(2, 4, 3, 3, 3).astype(np.float32),
        "strides": [1, 1, 1, 1, 1],
        "padding": "VALID",
        "data_format": "NCDHW",
        "dilations": [1, 1, 1, 1, 1],
        "name": "conv3d_backprop_filter_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: NCDHW, float32, SAME padding, strides=2, dilations=1
    input_dict_8 = {
        "input": np.random.randn(1, 2, 4, 4, 4).astype(np.float32),
        "filter_sizes": np.array([2, 2, 2, 2, 3], dtype=np.int32),
        "out_backprop": np.random.randn(1, 3, 2, 2, 2).astype(np.float32),
        "strides": [1, 1, 2, 2, 2],
        "padding": "SAME",
        "data_format": "NCDHW",
        "dilations": [1, 1, 1, 1, 1],
        "name": "conv3d_backprop_filter_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: NDHWC, float32, VALID padding, strides=1, dilations=2
    input_dict_9 = {
        "input": np.random.randn(2, 7, 7, 7, 2).astype(np.float32),
        "filter_sizes": np.array([3, 3, 3, 2, 2], dtype=np.int32),
        "out_backprop": np.random.randn(2, 3, 3, 3, 2).astype(np.float32),
        "strides": [1, 1, 1, 1, 1],
        "padding": "VALID",
        "data_format": "NDHWC",
        "dilations": [1, 2, 2, 2, 1],
        "name": "conv3d_backprop_filter_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: NCDHW, float32, VALID padding, strides=1, dilations=2
    input_dict_10 = {
        "input": np.random.randn(2, 2, 7, 7, 7).astype(np.float32),
        "filter_sizes": np.array([3, 3, 3, 2, 2], dtype=np.int32),
        "out_backprop": np.random.randn(2, 2, 3, 3, 3).astype(np.float32),
        "strides": [1, 1, 1, 1, 1],
        "padding": "VALID",
        "data_format": "NCDHW",
        "dilations": [1, 1, 2, 2, 2],
        "name": "conv3d_backprop_filter_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.Conv3DBackpropFilterV2"] = tf_raw_ops_Conv3DBackpropFilterV2_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DepthwiseConv2dNative_inputs():
    list_of_inputs = []

    # Input 1
    input_val = np.random.uniform(-1.0, 1.0, (1, 5, 5, 2)).astype(np.float32)
    filter_val = np.random.uniform(-1.0, 1.0, (3, 3, 2, 2)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter': filter_val,
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_val = np.random.uniform(-1.0, 1.0, (2, 10, 10, 3)).astype(np.float64)
    filter_val = np.random.uniform(-1.0, 1.0, (5, 5, 3, 1)).astype(np.float64)
    input_dict = {
        'input': input_val,
        'filter': filter_val,
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_val = np.random.uniform(-1.0, 1.0, (1, 8, 8, 1)).astype(np.float32)
    filter_val = np.random.uniform(-1.0, 1.0, (2, 2, 1, 3)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter': filter_val,
        'strides': [1, 1, 1, 1],
        'padding': 'EXPLICIT',
        'explicit_paddings': [0, 0, 1, 1, 1, 1, 0, 0],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_val = np.random.uniform(-1.0, 1.0, (2, 8, 8, 3)).astype(np.float32)
    filter_val = np.random.uniform(-1.0, 1.0, (3, 3, 3, 2)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter': filter_val,
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_val = np.random.uniform(-1.0, 1.0, (1, 4, 4, 4)).astype(np.float16)
    filter_val = np.random.uniform(-1.0, 1.0, (2, 2, 4, 1)).astype(np.float16)
    input_dict = {
        'input': input_val,
        'filter': filter_val,
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_val = np.random.uniform(-1.0, 1.0, (4, 6, 6, 2)).astype(np.float32)
    filter_val = np.random.uniform(-1.0, 1.0, (3, 3, 2, 3)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter': filter_val,
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_val = np.random.uniform(-1.0, 1.0, (1, 12, 12, 3)).astype(np.float32)
    filter_val = np.random.uniform(-1.0, 1.0, (5, 5, 3, 2)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter': filter_val,
        'strides': [1, 3, 3, 1],
        'padding': 'SAME',
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_val = np.random.uniform(-1.0, 1.0, (2, 8, 8, 2)).astype(np.float64)
    filter_val = np.random.uniform(-1.0, 1.0, (3, 3, 2, 2)).astype(np.float64)
    input_dict = {
        'input': input_val,
        'filter': filter_val,
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_val = np.random.uniform(-1.0, 1.0, (1, 3, 3, 1)).astype(np.float32)
    filter_val = np.random.uniform(-1.0, 1.0, (1, 1, 1, 1)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter': filter_val,
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'explicit_paddings': [],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_val = np.random.uniform(-1.0, 1.0, (2, 14, 14, 2)).astype(np.float32)
    filter_val = np.random.uniform(-1.0, 1.0, (3, 3, 2, 4)).astype(np.float32)
    input_dict = {
        'input': input_val,
        'filter': filter_val,
        'strides': [1, 2, 2, 1],
        'padding': 'EXPLICIT',
        'explicit_paddings': [0, 0, 2, 2, 2, 2, 0, 0],
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'conv_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.DepthwiseConv2dNative"] = tf_raw_ops_DepthwiseConv2dNative_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DepthwiseConv2dNativeBackpropFilter_inputs():
    list_of_inputs = []

    # Case 1: Standard valid case with float32, NHWC, VALID padding
    input_dict_1 = {
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv_case_1",
        'input': np.random.randn(1, 3, 3, 2).astype(np.float32),
        'filter_sizes': np.array([2, 2, 2, 1], dtype=np.int32),
        'out_backprop': np.random.randn(1, 2, 2, 2).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: SAME padding with depth multiplier of 2
    input_dict_2 = {
        'strides': [1, 1, 1, 1],
        'padding': "SAME",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv_case_2",
        'input': np.random.randn(2, 4, 4, 3).astype(np.float32),
        'filter_sizes': np.array([3, 3, 3, 2], dtype=np.int32),
        'out_backprop': np.random.randn(2, 4, 4, 6).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: VALID padding with strides of 2
    input_dict_3 = {
        'strides': [1, 2, 2, 1],
        'padding': "VALID",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv_case_3",
        'input': np.random.randn(1, 5, 5, 1).astype(np.float32),
        'filter_sizes': np.array([3, 3, 1, 3], dtype=np.int32),
        'out_backprop': np.random.randn(1, 2, 2, 3).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: Larger spatial dimensions, SAME padding
    input_dict_4 = {
        'strides': [1, 2, 2, 1],
        'padding': "SAME",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv_case_4",
        'input': np.random.randn(1, 8, 8, 4).astype(np.float32),
        'filter_sizes': np.array([3, 3, 4, 1], dtype=np.int32),
        'out_backprop': np.random.randn(1, 4, 4, 4).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: Depth multiplier of 1, larger filter size, VALID padding
    input_dict_5 = {
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv_case_5",
        'input': np.random.randn(4, 16, 16, 3).astype(np.float32),
        'filter_sizes': np.array([5, 5, 3, 1], dtype=np.int32),
        'out_backprop': np.random.randn(4, 12, 12, 3).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: Depth multiplier of 2, SAME padding
    input_dict_6 = {
        'strides': [1, 1, 1, 1],
        'padding': "SAME",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv_case_6",
        'input': np.random.randn(2, 10, 10, 2).astype(np.float32),
        'filter_sizes': np.array([3, 3, 2, 2], dtype=np.int32),
        'out_backprop': np.random.randn(2, 10, 10, 4).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: Strided SAME padding with single channel
    input_dict_7 = {
        'strides': [1, 2, 2, 1],
        'padding': "SAME",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv_case_7",
        'input': np.random.randn(1, 4, 4, 1).astype(np.float32),
        'filter_sizes': np.array([2, 2, 1, 2], dtype=np.int32),
        'out_backprop': np.random.randn(1, 2, 2, 2).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: float64 dtype support
    input_dict_8 = {
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv_case_8",
        'input': np.random.randn(1, 3, 3, 2).astype(np.float64),
        'filter_sizes': np.array([2, 2, 2, 1], dtype=np.int32),
        'out_backprop': np.random.randn(1, 2, 2, 2).astype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: Uniform negative inputs
    input_dict_9 = {
        'strides': [1, 1, 1, 1],
        'padding': "VALID",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv_case_9",
        'input': np.random.uniform(-1.0, 1.0, (2, 3, 3, 2)).astype(np.float32),
        'filter_sizes': np.array([2, 2, 2, 2], dtype=np.int32),
        'out_backprop': np.random.uniform(-1.0, 1.0, (2, 2, 2, 4)).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: Strided SAME padding with depthwise_multiplier of 2
    input_dict_10 = {
        'strides': [1, 2, 2, 1],
        'padding': "SAME",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'dilations': [1, 1, 1, 1],
        'name': "conv_case_10",
        'input': np.random.randn(1, 6, 6, 2).astype(np.float32),
        'filter_sizes': np.array([3, 3, 2, 2], dtype=np.int32),
        'out_backprop': np.random.randn(1, 3, 3, 4).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.DepthwiseConv2dNativeBackpropFilter"] = tf_raw_ops_DepthwiseConv2dNativeBackpropFilter_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_dilation2d_backprop_filter_inputs():
    list_of_inputs = []
    
    # Case 1: Float32, strides=[1,1,1,1], rates=[1,1,1,1], VALID padding
    input_val = np.random.randn(1, 3, 3, 1).astype(np.float32)
    filter_val = np.random.randn(2, 2, 1).astype(np.float32)
    out_backprop_val = np.random.randn(1, 2, 2, 1).astype(np.float32)
    list_of_inputs.append({
        'name': 'case1',
        'input': input_val,
        'filter': filter_val,
        'out_backprop': out_backprop_val,
        'strides': [1, 1, 1, 1],
        'rates': [1, 1, 1, 1],
        'padding': 'VALID'
    })

    # Case 2: Float64, strides=[1,1,1,1], rates=[1,1,1,1], SAME padding
    input_val = np.random.randn(1, 3, 3, 1).astype(np.float64)
    filter_val = np.random.randn(2, 2, 1).astype(np.float64)
    out_backprop_val = np.random.randn(1, 3, 3, 1).astype(np.float64)
    list_of_inputs.append({
        'name': 'case2',
        'input': input_val,
        'filter': filter_val,
        'out_backprop': out_backprop_val,
        'strides': [1, 1, 1, 1],
        'rates': [1, 1, 1, 1],
        'padding': 'SAME'
    })

    # Case 3: Int32, strides=[1,2,2,1], rates=[1,1,1,1], VALID padding
    input_val = np.random.randint(-10, 10, size=(2, 5, 5, 2)).astype(np.int32)
    filter_val = np.random.randint(-10, 10, size=(3, 3, 2)).astype(np.int32)
    out_backprop_val = np.random.randint(-10, 10, size=(2, 2, 2, 2)).astype(np.int32)
    list_of_inputs.append({
        'name': 'case3',
        'input': input_val,
        'filter': filter_val,
        'out_backprop': out_backprop_val,
        'strides': [1, 2, 2, 1],
        'rates': [1, 1, 1, 1],
        'padding': 'VALID'
    })

    # Case 4: Float32 with larger batch/channel, strides=[1,2,2,1], rates=[1,1,1,1], SAME padding
    input_val = np.random.randn(2, 5, 5, 2).astype(np.float32)
    filter_val = np.random.randn(3, 3, 2).astype(np.float32)
    out_backprop_val = np.random.randn(2, 3, 3, 2).astype(np.float32)
    list_of_inputs.append({
        'name': 'case4',
        'input': input_val,
        'filter': filter_val,
        'out_backprop': out_backprop_val,
        'strides': [1, 2, 2, 1],
        'rates': [1, 1, 1, 1],
        'padding': 'SAME'
    })

    # Case 5: Float32, strides=[1,1,1,1], rates=[1,2,2,1], VALID padding
    input_val = np.random.randn(1, 5, 5, 1).astype(np.float32)
    filter_val = np.random.randn(2, 2, 1).astype(np.float32)
    out_backprop_val = np.random.randn(1, 3, 3, 1).astype(np.float32)
    list_of_inputs.append({
        'name': 'case5',
        'input': input_val,
        'filter': filter_val,
        'out_backprop': out_backprop_val,
        'strides': [1, 1, 1, 1],
        'rates': [1, 2, 2, 1],
        'padding': 'VALID'
    })

    # Case 6: Float64, strides=[1,1,1,1], rates=[1,2,2,1], SAME padding
    input_val = np.random.randn(1, 5, 5, 1).astype(np.float64)
    filter_val = np.random.randn(2, 2, 1).astype(np.float64)
    out_backprop_val = np.random.randn(1, 5, 5, 1).astype(np.float64)
    list_of_inputs.append({
        'name': 'case6',
        'input': input_val,
        'filter': filter_val,
        'out_backprop': out_backprop_val,
        'strides': [1, 1, 1, 1],
        'rates': [1, 2, 2, 1],
        'padding': 'SAME'
    })

    # Case 7: Float32 with 3 channels, strides=[1,1,1,1], rates=[1,1,1,1], VALID padding
    input_val = np.random.randn(4, 4, 4, 3).astype(np.float32)
    filter_val = np.random.randn(2, 2, 3).astype(np.float32)
    out_backprop_val = np.random.randn(4, 3, 3, 3).astype(np.float32)
    list_of_inputs.append({
        'name': 'case7',
        'input': input_val,
        'filter': filter_val,
        'out_backprop': out_backprop_val,
        'strides': [1, 1, 1, 1],
        'rates': [1, 1, 1, 1],
        'padding': 'VALID'
    })

    # Case 8: Int64, strides=[1,1,2,1], rates=[1,1,1,1], SAME padding
    input_val = np.random.randint(-100, 100, size=(2, 4, 4, 2)).astype(np.int64)
    filter_val = np.random.randint(-100, 100, size=(2, 2, 2)).astype(np.int64)
    out_backprop_val = np.random.randint(-100, 100, size=(2, 4, 2, 2)).astype(np.int64)
    list_of_inputs.append({
        'name': 'case8',
        'input': input_val,
        'filter': filter_val,
        'out_backprop': out_backprop_val,
        'strides': [1, 1, 2, 1],
        'rates': [1, 1, 1, 1],
        'padding': 'SAME'
    })

    # Case 9: Float32, strides=[1,3,3,1], rates=[1,1,1,1], VALID padding
    input_val = np.random.randn(1, 7, 7, 1).astype(np.float32)
    filter_val = np.random.randn(3, 3, 1).astype(np.float32)
    out_backprop_val = np.random.randn(1, 2, 2, 1).astype(np.float32)
    list_of_inputs.append({
        'name': 'case9',
        'input': input_val,
        'filter': filter_val,
        'out_backprop': out_backprop_val,
        'strides': [1, 3, 3, 1],
        'rates': [1, 1, 1, 1],
        'padding': 'VALID'
    })

    # Case 10: Float32 with unequal stride and rate, SAME padding
    input_val = np.random.randn(2, 6, 4, 2).astype(np.float32)
    filter_val = np.random.randn(2, 3, 2).astype(np.float32)
    out_backprop_val = np.random.randn(2, 2, 4, 2).astype(np.float32)
    list_of_inputs.append({
        'name': 'case10',
        'input': input_val,
        'filter': filter_val,
        'out_backprop': out_backprop_val,
        'strides': [1, 3, 1, 1],
        'rates': [1, 2, 1, 1],
        'padding': 'SAME'
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.Dilation2DBackpropFilter"] = tf_dilation2d_backprop_filter_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_dilation2d_backprop_input_inputs():
    list_of_inputs = []

    # Input 1: float32, VALID, strides [1, 1, 1, 1]
    input_dict = {
        "input": np.random.randn(1, 5, 5, 1).astype(np.float32),
        "filter": np.random.randn(3, 3, 1).astype(np.float32),
        "out_backprop": np.random.randn(1, 3, 3, 1).astype(np.float32),
        "strides": [1, 1, 1, 1],
        "rates": [1, 1, 1, 1],
        "padding": "VALID",
        "name": "test_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, SAME, strides [1, 1, 1, 1]
    input_dict = {
        "input": np.random.randn(2, 4, 4, 3).astype(np.float64),
        "filter": np.random.randn(2, 2, 3).astype(np.float64),
        "out_backprop": np.random.randn(2, 4, 4, 3).astype(np.float64),
        "strides": [1, 1, 1, 1],
        "rates": [1, 1, 1, 1],
        "padding": "SAME",
        "name": "test_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32, VALID, strides [1, 2, 2, 1]
    input_dict = {
        "input": np.random.randint(-10, 10, size=(1, 6, 6, 2)).astype(np.int32),
        "filter": np.random.randint(-5, 5, size=(2, 2, 2)).astype(np.int32),
        "out_backprop": np.random.randint(-10, 10, size=(1, 3, 3, 2)).astype(np.int32),
        "strides": [1, 2, 2, 1],
        "rates": [1, 1, 1, 1],
        "padding": "VALID",
        "name": "test_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int32, SAME, strides [1, 2, 2, 1]
    input_dict = {
        "input": np.random.randint(-5, 5, size=(1, 5, 5, 1)).astype(np.int32),
        "filter": np.random.randint(-3, 3, size=(3, 3, 1)).astype(np.int32),
        "out_backprop": np.random.randint(-5, 5, size=(1, 3, 3, 1)).astype(np.int32),
        "strides": [1, 2, 2, 1],
        "rates": [1, 1, 1, 1],
        "padding": "SAME",
        "name": "test_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, VALID, strides [1, 1, 1, 1], rates [1, 2, 2, 1]
    input_dict = {
        "input": np.random.randn(1, 7, 7, 1).astype(np.float32),
        "filter": np.random.randn(3, 3, 1).astype(np.float32),
        "out_backprop": np.random.randn(1, 3, 3, 1).astype(np.float32),
        "strides": [1, 1, 1, 1],
        "rates": [1, 2, 2, 1],
        "padding": "VALID",
        "name": "test_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float32, SAME, strides [1, 1, 1, 1], rates [1, 2, 2, 1]
    input_dict = {
        "input": np.random.randn(1, 5, 5, 1).astype(np.float32),
        "filter": np.random.randn(3, 3, 1).astype(np.float32),
        "out_backprop": np.random.randn(1, 5, 5, 1).astype(np.float32),
        "strides": [1, 1, 1, 1],
        "rates": [1, 2, 2, 1],
        "padding": "SAME",
        "name": "test_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int32, VALID, strides [1, 1, 1, 1]
    input_dict = {
        "input": np.random.randint(-128, 127, size=(1, 10, 10, 4)).astype(np.int32),
        "filter": np.random.randint(-5, 5, size=(5, 5, 4)).astype(np.int32),
        "out_backprop": np.random.randint(-128, 127, size=(1, 6, 6, 4)).astype(np.int32),
        "strides": [1, 1, 1, 1],
        "rates": [1, 1, 1, 1],
        "padding": "VALID",
        "name": "test_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int64, SAME, strides [1, 3, 3, 1]
    input_dict = {
        "input": np.random.randint(-1000, 1000, size=(1, 9, 9, 1)).astype(np.int64),
        "filter": np.random.randint(-100, 100, size=(3, 3, 1)).astype(np.int64),
        "out_backprop": np.random.randint(-1000, 1000, size=(1, 3, 3, 1)).astype(np.int64),
        "strides": [1, 3, 3, 1],
        "rates": [1, 1, 1, 1],
        "padding": "SAME",
        "name": "test_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64, VALID, strides [1, 2, 2, 1], rates [1, 2, 2, 1]
    input_dict = {
        "input": np.random.randn(2, 10, 10, 2).astype(np.float64),
        "filter": np.random.randn(3, 3, 2).astype(np.float64),
        "out_backprop": np.random.randn(2, 3, 3, 2).astype(np.float64),
        "strides": [1, 2, 2, 1],
        "rates": [1, 2, 2, 1],
        "padding": "VALID",
        "name": "test_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, SAME, strides [1, 2, 2, 1], rates [1, 2, 2, 1]
    input_dict = {
        "input": np.random.randn(1, 6, 6, 1).astype(np.float32),
        "filter": np.random.randn(3, 3, 1).astype(np.float32),
        "out_backprop": np.random.randn(1, 3, 3, 1).astype(np.float32),
        "strides": [1, 2, 2, 1],
        "rates": [1, 2, 2, 1],
        "padding": "SAME",
        "name": "test_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Dilation2DBackpropInput"] = tf_dilation2d_backprop_input_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_ExtractImagePatches_inputs():
    list_of_inputs = []

    # Input 1
    images = np.random.randn(1, 4, 4, 1).astype(np.float32)
    ksizes = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "patch_1"
    list_of_inputs.append({
        "images": images,
        "ksizes": ksizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    })

    # Input 2
    images = np.random.randn(2, 8, 8, 3).astype(np.float32)
    ksizes = [1, 3, 3, 1]
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"
    name = "patch_2"
    list_of_inputs.append({
        "images": images,
        "ksizes": ksizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    })

    # Input 3
    images = np.random.randint(0, 10, size=(1, 10, 10, 1)).astype(np.int32)
    ksizes = [1, 4, 4, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 2, 2, 1]
    padding = "VALID"
    name = "patch_3"
    list_of_inputs.append({
        "images": images,
        "ksizes": ksizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    })

    # Input 4
    images = np.random.randn(4, 16, 16, 2).astype(np.float16)
    ksizes = [1, 2, 3, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"
    name = "patch_4"
    list_of_inputs.append({
        "images": images,
        "ksizes": ksizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    })

    # Input 5
    images = np.random.randint(0, 255, size=(1, 5, 5, 3)).astype(np.uint8)
    ksizes = [1, 3, 3, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "patch_5"
    list_of_inputs.append({
        "images": images,
        "ksizes": ksizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    })

    # Input 6
    images = np.random.randn(1, 12, 12, 1).astype(np.float64)
    ksizes = [1, 5, 5, 1]
    strides = [1, 3, 3, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"
    name = "patch_6"
    list_of_inputs.append({
        "images": images,
        "ksizes": ksizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    })

    # Input 7
    images = np.random.randint(-128, 127, size=(1, 6, 6, 1)).astype(np.int8)
    ksizes = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "patch_7"
    list_of_inputs.append({
        "images": images,
        "ksizes": ksizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    })

    # Input 8
    images = np.random.randint(-100, 100, size=(2, 4, 4, 4)).astype(np.int16)
    ksizes = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 2, 2, 1]
    padding = "SAME"
    name = "patch_8"
    list_of_inputs.append({
        "images": images,
        "ksizes": ksizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    })

    # Input 9
    images = (np.random.randn(1, 3, 3, 1) + 1j * np.random.randn(1, 3, 3, 1)).astype(np.complex64)
    ksizes = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "patch_9"
    list_of_inputs.append({
        "images": images,
        "ksizes": ksizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    })

    # Input 10
    images = np.random.randn(1, 14, 14, 3).astype(np.float32)
    ksizes = [1, 3, 3, 1]
    strides = [1, 3, 3, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "patch_10"
    list_of_inputs.append({
        "images": images,
        "ksizes": ksizes,
        "strides": strides,
        "rates": rates,
        "padding": padding,
        "name": name
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.ExtractImagePatches"] = tf_raw_ops_ExtractImagePatches_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ExtractVolumePatches_inputs():
    list_of_inputs = []

    # Input 1
    input_val = np.random.rand(1, 2, 2, 2, 1).astype(np.float32)
    list_of_inputs.append({
        'input': input_val,
        'ksizes': [1, 1, 1, 1, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': "VALID",
        'name': "extract_1"
    })

    # Input 2
    input_val = np.random.randint(0, 10, size=(2, 3, 3, 3, 2)).astype(np.int32)
    list_of_inputs.append({
        'input': input_val,
        'ksizes': [1, 2, 2, 2, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': "SAME",
        'name': "extract_2"
    })

    # Input 3
    input_val = np.random.rand(1, 4, 4, 4, 3).astype(np.float64)
    list_of_inputs.append({
        'input': input_val,
        'ksizes': [1, 2, 2, 2, 1],
        'strides': [1, 2, 2, 2, 1],
        'padding': "VALID",
        'name': "extract_3"
    })

    # Input 4
    input_val = np.random.randint(0, 256, size=(1, 5, 5, 5, 1)).astype(np.uint8)
    list_of_inputs.append({
        'input': input_val,
        'ksizes': [1, 3, 3, 3, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': "SAME",
        'name': "extract_4"
    })

    # Input 5
    input_val = np.random.randint(-100, 100, size=(3, 2, 2, 2, 4)).astype(np.int64)
    list_of_inputs.append({
        'input': input_val,
        'ksizes': [1, 1, 1, 1, 1],
        'strides': [1, 2, 2, 2, 1],
        'padding': "VALID",
        'name': "extract_5"
    })

    # Input 6
    input_val = np.random.rand(2, 3, 4, 5, 1).astype(np.float32)
    list_of_inputs.append({
        'input': input_val,
        'ksizes': [1, 2, 2, 2, 1],
        'strides': [1, 1, 2, 2, 1],
        'padding': "SAME",
        'name': "extract_6"
    })

    # Input 7
    input_val = np.random.randint(-10, 10, size=(1, 2, 2, 2, 2)).astype(np.int32)
    list_of_inputs.append({
        'input': input_val,
        'ksizes': [1, 2, 2, 2, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': "VALID",
        'name': "extract_7"
    })

    # Input 8
    input_val = np.random.rand(1, 3, 3, 3, 1).astype(np.float64)
    list_of_inputs.append({
        'input': input_val,
        'ksizes': [1, 3, 1, 1, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': "VALID",
        'name': "extract_8"
    })

    # Input 9
    input_val = np.random.randint(0, 256, size=(2, 2, 3, 2, 1)).astype(np.uint8)
    list_of_inputs.append({
        'input': input_val,
        'ksizes': [1, 1, 2, 1, 1],
        'strides': [1, 1, 1, 1, 1],
        'padding': "SAME",
        'name': "extract_9"
    })

    # Input 10
    input_val = np.random.randint(-100, 100, size=(1, 4, 2, 4, 3)).astype(np.int64)
    list_of_inputs.append({
        'input': input_val,
        'ksizes': [1, 2, 1, 2, 1],
        'strides': [1, 2, 1, 2, 1],
        'padding': "VALID",
        'name': "extract_10"
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.ExtractVolumePatches"] = tf_raw_ops_ExtractVolumePatches_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FusedPadConv2D_inputs():
    list_of_inputs = []
    
    # Input 1
    input_val = np.random.randn(1, 5, 5, 3).astype(np.float32)
    paddings_val = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32)
    filter_val = np.random.randn(3, 3, 3, 2).astype(np.float32)
    input_dict = {
        'mode': 'REFLECT',
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'name': 'fused_pad_conv_1',
        'input': input_val,
        'paddings': paddings_val,
        'filter': filter_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_val = np.random.randn(1, 5, 5, 3).astype(np.float32)
    paddings_val = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32)
    filter_val = np.random.randn(3, 3, 3, 2).astype(np.float32)
    input_dict = {
        'mode': 'SYMMETRIC',
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'name': 'fused_pad_conv_2',
        'input': input_val,
        'paddings': paddings_val,
        'filter': filter_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_val = np.random.randn(2, 10, 10, 4).astype(np.float64)
    paddings_val = np.array([[0, 0], [2, 2], [2, 2], [0, 0]], dtype=np.int32)
    filter_val = np.random.randn(5, 5, 4, 8).astype(np.float64)
    input_dict = {
        'mode': 'REFLECT',
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'name': 'fused_pad_conv_3',
        'input': input_val,
        'paddings': paddings_val,
        'filter': filter_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_val = np.random.randn(1, 6, 6, 1).astype(np.float16)
    paddings_val = np.array([[0, 0], [1, 2], [1, 2], [0, 0]], dtype=np.int32)
    filter_val = np.random.randn(2, 2, 1, 2).astype(np.float16)
    input_dict = {
        'mode': 'SYMMETRIC',
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'name': 'fused_pad_conv_4',
        'input': input_val,
        'paddings': paddings_val,
        'filter': filter_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_val = np.random.randn(4, 8, 8, 2).astype(np.float32)
    paddings_val = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32)
    filter_val = np.random.randn(1, 1, 2, 4).astype(np.float32)
    input_dict = {
        'mode': 'REFLECT',
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'name': 'fused_pad_conv_5',
        'input': input_val,
        'paddings': paddings_val,
        'filter': filter_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_val = np.random.randn(2, 15, 15, 3).astype(np.float32)
    paddings_val = np.array([[0, 0], [3, 3], [3, 3], [0, 0]], dtype=np.int32)
    filter_val = np.random.randn(7, 7, 3, 5).astype(np.float32)
    input_dict = {
        'mode': 'SYMMETRIC',
        'strides': [1, 3, 3, 1],
        'padding': 'VALID',
        'name': 'fused_pad_conv_6',
        'input': input_val,
        'paddings': paddings_val,
        'filter': filter_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_val = np.random.randn(1, 4, 4, 8).astype(np.float16)
    paddings_val = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32)
    filter_val = np.random.randn(3, 3, 8, 16).astype(np.float16)
    input_dict = {
        'mode': 'REFLECT',
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'name': 'fused_pad_conv_7',
        'input': input_val,
        'paddings': paddings_val,
        'filter': filter_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_val = np.random.randn(1, 3, 3, 1).astype(np.float64)
    paddings_val = np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32)
    filter_val = np.random.randn(3, 3, 1, 1).astype(np.float64)
    input_dict = {
        'mode': 'SYMMETRIC',
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'name': 'fused_pad_conv_8',
        'input': input_val,
        'paddings': paddings_val,
        'filter': filter_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_val = np.random.randn(3, 12, 12, 4).astype(np.float32)
    paddings_val = np.array([[0, 0], [2, 1], [2, 1], [0, 0]], dtype=np.int32)
    filter_val = np.random.randn(4, 4, 4, 2).astype(np.float32)
    input_dict = {
        'mode': 'REFLECT',
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'name': 'fused_pad_conv_9',
        'input': input_val,
        'paddings': paddings_val,
        'filter': filter_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_val = np.random.randn(1, 20, 20, 3).astype(np.float32)
    paddings_val = np.array([[0, 0], [4, 4], [4, 4], [0, 0]], dtype=np.int32)
    filter_val = np.random.randn(5, 5, 3, 3).astype(np.float32)
    input_dict = {
        'mode': 'SYMMETRIC',
        'strides': [1, 2, 2, 1],
        'padding': 'SAME',
        'name': 'fused_pad_conv_10',
        'input': input_val,
        'paddings': paddings_val,
        'filter': filter_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.FusedPadConv2D"] = tf_raw_ops_FusedPadConv2D_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FusedResizeAndPadConv2D_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'resize_align_corners': False,
        'name': "op1",
        'input': np.random.rand(2, 8, 8, 3).astype(np.float32),
        'size': np.array([4, 4], dtype=np.int32),
        'paddings': np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(3, 3, 3, 2).astype(np.float32),
        'mode': "REFLECT",
        'strides': [1, 1, 1, 1],
        'padding': "VALID"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'resize_align_corners': True,
        'name': "op2",
        'input': np.random.rand(1, 10, 10, 1).astype(np.float32),
        'size': np.array([5, 5], dtype=np.int32),
        'paddings': np.array([[0, 0], [2, 2], [2, 2], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(2, 2, 1, 4).astype(np.float32),
        'mode': "SYMMETRIC",
        'strides': [1, 2, 2, 1],
        'padding': "SAME"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'resize_align_corners': False,
        'name': "op3",
        'input': np.random.rand(3, 6, 6, 2).astype(np.float64),
        'size': np.array([3, 3], dtype=np.int32),
        'paddings': np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(2, 2, 2, 1).astype(np.float64),
        'mode': "REFLECT",
        'strides': [1, 1, 1, 1],
        'padding': "SAME"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'resize_align_corners': True,
        'name': "op4",
        'input': np.random.rand(1, 12, 12, 4).astype(np.float16),
        'size': np.array([6, 6], dtype=np.int32),
        'paddings': np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(3, 3, 4, 2).astype(np.float16),
        'mode': "REFLECT",
        'strides': [1, 1, 1, 1],
        'padding': "VALID"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'resize_align_corners': False,
        'name': "op5",
        'input': np.random.rand(2, 16, 16, 3).astype(np.float32),
        'size': np.array([8, 8], dtype=np.int32),
        'paddings': np.array([[0, 0], [2, 1], [1, 2], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(3, 3, 3, 8).astype(np.float32),
        'mode': "SYMMETRIC",
        'strides': [1, 1, 1, 1],
        'padding': "VALID"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'resize_align_corners': True,
        'name': "op6",
        'input': np.random.rand(1, 5, 5, 2).astype(np.float32),
        'size': np.array([10, 10], dtype=np.int32),
        'paddings': np.array([[0, 0], [2, 2], [2, 2], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(5, 5, 2, 2).astype(np.float32),
        'mode': "REFLECT",
        'strides': [1, 1, 1, 1],
        'padding': "SAME"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'resize_align_corners': False,
        'name': "op7",
        'input': np.random.rand(4, 8, 8, 1).astype(np.float64),
        'size': np.array([2, 2], dtype=np.int32),
        'paddings': np.array([[0, 0], [1, 1], [1, 1], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(1, 1, 1, 1).astype(np.float64),
        'mode': "REFLECT",
        'strides': [1, 1, 1, 1],
        'padding': "VALID"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'resize_align_corners': False,
        'name': "op8",
        'input': np.random.rand(1, 4, 4, 3).astype(np.float32),
        'size': np.array([3, 5], dtype=np.int32),
        'paddings': np.array([[0, 0], [1, 1], [2, 2], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(3, 3, 3, 4).astype(np.float32),
        'mode': "SYMMETRIC",
        'strides': [1, 1, 1, 1],
        'padding': "SAME"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'resize_align_corners': True,
        'name': "op9",
        'input': np.random.rand(2, 14, 14, 2).astype(np.float16),
        'size': np.array([7, 7], dtype=np.int32),
        'paddings': np.array([[0, 0], [3, 3], [3, 3], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(4, 4, 2, 4).astype(np.float16),
        'mode': "SYMMETRIC",
        'strides': [1, 2, 2, 1],
        'padding': "VALID"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'resize_align_corners': True,
        'name': "op10",
        'input': np.random.rand(1, 20, 20, 1).astype(np.float32),
        'size': np.array([5, 10], dtype=np.int32),
        'paddings': np.array([[0, 0], [1, 2], [3, 4], [0, 0]], dtype=np.int32),
        'filter': np.random.rand(2, 2, 1, 1).astype(np.float32),
        'mode': "REFLECT",
        'strides': [1, 1, 1, 1],
        'padding': "SAME"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.FusedResizeAndPadConv2D"] = tf_raw_ops_FusedResizeAndPadConv2D_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MaxPool_inputs():
    list_of_inputs = []

    # Input 1, float32, NHWC, VALID
    list_of_inputs.append({
        'input': np.random.randn(1, 4, 4, 1).astype(np.float32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': "VALID",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'name': "maxpool_1"
    })

    # Input 2, float64, NHWC, SAME
    list_of_inputs.append({
        'input': np.random.randn(2, 8, 8, 3).astype(np.float64),
        'ksize': [1, 3, 3, 1],
        'strides': [1, 1, 1, 1],
        'padding': "SAME",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'name': "maxpool_2"
    })

    # Input 3, float32, NHWC, VALID
    list_of_inputs.append({
        'input': np.random.randn(1, 10, 10, 2).astype(np.float32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': "VALID",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'name': "maxpool_3"
    })

    # Input 4, int32, NHWC, SAME
    list_of_inputs.append({
        'input': np.random.randint(-50, 50, (1, 6, 6, 2)).astype(np.int32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': "SAME",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'name': "maxpool_4"
    })

    # Input 5, int32, NHWC, VALID
    list_of_inputs.append({
        'input': np.random.randint(0, 255, (3, 16, 16, 4)).astype(np.int32),
        'ksize': [1, 4, 4, 1],
        'strides': [1, 4, 4, 1],
        'padding': "VALID",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'name': "maxpool_5"
    })

    # Input 6, int32, NHWC, SAME
    list_of_inputs.append({
        'input': np.random.randint(-100, 100, (1, 5, 5, 3)).astype(np.int32),
        'ksize': [1, 3, 3, 1],
        'strides': [1, 1, 1, 1],
        'padding': "SAME",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'name': "maxpool_6"
    })

    # Input 7, int32, NHWC, EXPLICIT
    list_of_inputs.append({
        'input': np.random.randint(-10, 10, (1, 4, 4, 1)).astype(np.int32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': "EXPLICIT",
        'explicit_paddings': [0, 0, 1, 1, 1, 1, 0, 0],
        'data_format': "NHWC",
        'name': "maxpool_7"
    })

    # Input 8, float32, NHWC, EXPLICIT
    list_of_inputs.append({
        'input': np.random.randn(1, 14, 14, 1).astype(np.float32),
        'ksize': [1, 3, 3, 1],
        'strides': [1, 2, 2, 1],
        'padding': "EXPLICIT",
        'explicit_paddings': [0, 0, 1, 1, 2, 2, 0, 0],
        'data_format': "NHWC",
        'name': "maxpool_8"
    })

    # Input 9, int32, NHWC, SAME
    list_of_inputs.append({
        'input': np.random.randint(0, 1000, (2, 32, 32, 1)).astype(np.int32),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': "SAME",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'name': "maxpool_9"
    })

    # Input 10, float64, NHWC, VALID
    list_of_inputs.append({
        'input': np.random.randn(4, 8, 8, 3).astype(np.float64),
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': "VALID",
        'explicit_paddings': [],
        'data_format': "NHWC",
        'name': "maxpool_10"
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.MaxPool"] = tf_raw_ops_MaxPool_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_maxpool3dgrad_inputs():
    list_of_inputs = []

    # 1. Float32, NDHWC, VALID, simple 2x2x2
    orig_input = np.arange(8, dtype=np.float32).reshape((1, 2, 2, 2, 1))
    orig_output = np.array([[[[[7.0]]]]], dtype=np.float32)
    grad = np.array([[[[[1.0]]]]], dtype=np.float32)
    ksize = [1, 2, 2, 2, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = "max_pool_3d_grad_1"
    
    list_of_inputs.append({
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': ksize,
        'strides': strides,
        'padding': padding,
        'data_format': data_format,
        'name': name
    })

    # 2. Float32, NDHWC, SAME
    orig_input = np.arange(8, dtype=np.float32).reshape((1, 2, 2, 2, 1))
    orig_output = np.arange(8, dtype=np.float32).reshape((1, 2, 2, 2, 1))
    grad = np.ones((1, 2, 2, 2, 1), dtype=np.float32)
    ksize = [1, 2, 2, 2, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"
    data_format = "NDHWC"
    name = "max_pool_3d_grad_2"

    list_of_inputs.append({
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': ksize,
        'strides': strides,
        'padding': padding,
        'data_format': data_format,
        'name': name
    })

    # 3. Float32, Negative values, NDHWC, VALID
    orig_input = np.random.uniform(-10, 10, (1, 3, 3, 3, 2)).astype(np.float32)
    orig_output = np.random.uniform(-10, 10, (1, 2, 2, 2, 2)).astype(np.float32)
    grad = np.random.uniform(-1, 1, (1, 2, 2, 2, 2)).astype(np.float32)
    ksize = [1, 2, 2, 2, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = "max_pool_3d_grad_3"

    list_of_inputs.append({
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': ksize,
        'strides': strides,
        'padding': padding,
        'data_format': data_format,
        'name': name
    })

    # 4. Float32, NDHWC, VALID, stride 2
    orig_input = np.random.uniform(0, 5, (2, 4, 4, 4, 1)).astype(np.float32)
    orig_output = np.random.uniform(0, 5, (2, 2, 2, 2, 1)).astype(np.float32)
    grad = np.random.uniform(-1, 1, (2, 2, 2, 2, 1)).astype(np.float32)
    ksize = [1, 2, 2, 2, 1]
    strides = [1, 2, 2, 2, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = "max_pool_3d_grad_4"

    list_of_inputs.append({
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': ksize,
        'strides': strides,
        'padding': padding,
        'data_format': data_format,
        'name': name
    })

    # 5. Float32, NDHWC, SAME, ksize 3
    orig_input = np.random.uniform(-5, 5, (1, 5, 5, 5, 3)).astype(np.float32)
    orig_output = np.random.uniform(-5, 5, (1, 5, 5, 5, 3)).astype(np.float32)
    grad = np.random.uniform(-1, 1, (1, 5, 5, 5, 3)).astype(np.float32)
    ksize = [1, 3, 3, 3, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"
    data_format = "NDHWC"
    name = "max_pool_3d_grad_5"

    list_of_inputs.append({
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': ksize,
        'strides': strides,
        'padding': padding,
        'data_format': data_format,
        'name': name
    })

    # 6. Float32, NDHWC, SAME, stride 2
    orig_input = np.random.uniform(-2, 2, (1, 6, 6, 6, 1)).astype(np.float32)
    orig_output = np.random.uniform(-2, 2, (1, 3, 3, 3, 1)).astype(np.float32)
    grad = np.random.uniform(-1, 1, (1, 3, 3, 3, 1)).astype(np.float32)
    ksize = [1, 2, 2, 2, 1]
    strides = [1, 2, 2, 2, 1]
    padding = "SAME"
    data_format = "NDHWC"
    name = "max_pool_3d_grad_6"

    list_of_inputs.append({
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': ksize,
        'strides': strides,
        'padding': padding,
        'data_format': data_format,
        'name': name
    })

    # 7. Float32, NDHWC, VALID, ksize 3
    orig_input = np.random.uniform(-1, 1, (1, 4, 4, 4, 2)).astype(np.float32)
    orig_output = np.random.uniform(-1, 1, (1, 2, 2, 2, 2)).astype(np.float32)
    grad = np.random.uniform(-1, 1, (1, 2, 2, 2, 2)).astype(np.float32)
    ksize = [1, 3, 3, 3, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = "max_pool_3d_grad_7"

    list_of_inputs.append({
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': ksize,
        'strides': strides,
        'padding': padding,
        'data_format': data_format,
        'name': name
    })

    # 8. Float32, NDHWC, VALID, stride 2, larger input
    orig_input = np.random.uniform(-1, 1, (1, 8, 8, 8, 1)).astype(np.float32)
    orig_output = np.random.uniform(-1, 1, (1, 4, 4, 4, 1)).astype(np.float32)
    grad = np.random.uniform(-1, 1, (1, 4, 4, 4, 1)).astype(np.float32)
    ksize = [1, 2, 2, 2, 1]
    strides = [1, 2, 2, 2, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = "max_pool_3d_grad_8"

    list_of_inputs.append({
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': ksize,
        'strides': strides,
        'padding': padding,
        'data_format': data_format,
        'name': name
    })

    # 9. Float32, NDHWC, SAME, ksize 1, stride 1 (effectively identity)
    orig_input = np.random.uniform(-5, 5, (2, 3, 3, 3, 2)).astype(np.float32)
    orig_output = orig_input
    grad = np.random.uniform(-1, 1, (2, 3, 3, 3, 2)).astype(np.float32)
    ksize = [1, 1, 1, 1, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"
    data_format = "NDHWC"
    name = "max_pool_3d_grad_9"

    list_of_inputs.append({
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': ksize,
        'strides': strides,
        'padding': padding,
        'data_format': data_format,
        'name': name
    })

    # 10. Float32, NDHWC, VALID, non-symmetric input shape
    orig_input = np.random.uniform(-10, 10, (1, 4, 2, 2, 1)).astype(np.float32)
    orig_output = np.random.uniform(-10, 10, (1, 2, 1, 1, 1)).astype(np.float32)
    grad = np.random.uniform(-1, 1, (1, 2, 1, 1, 1)).astype(np.float32)
    ksize = [1, 2, 2, 2, 1]
    strides = [1, 2, 2, 2, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = "max_pool_3d_grad_10"

    list_of_inputs.append({
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': ksize,
        'strides': strides,
        'padding': padding,
        'data_format': data_format,
        'name': name
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.MaxPool3DGrad"] = tf_maxpool3dgrad_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MaxPoolGradGrad_inputs():
    list_of_inputs = []

    # Input 1: Basic NHWC float32 with VALID padding
    orig_input = np.ones((1, 4, 4, 1), dtype=np.float32)
    orig_output = np.ones((1, 2, 2, 1), dtype=np.float32)
    grad = np.random.randn(1, 4, 4, 1).astype(np.float32)
    input_dict = {
        'data_format': 'NHWC',
        'name': 'mp_grad_grad_1',
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: NHWC float32 with SAME padding and kernel size 3
    orig_input = np.ones((2, 4, 4, 3), dtype=np.float32)
    orig_output = np.ones((2, 4, 4, 3), dtype=np.float32)
    grad = np.random.randn(2, 4, 4, 3).astype(np.float32)
    input_dict = {
        'data_format': 'NHWC',
        'name': 'mp_grad_grad_2',
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': [1, 3, 3, 1],
        'strides': [1, 1, 1, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: NHWC float64 with VALID padding
    orig_input = np.ones((1, 6, 6, 2), dtype=np.float64)
    orig_output = np.ones((1, 3, 3, 2), dtype=np.float64)
    grad = np.random.randn(1, 6, 6, 2).astype(np.float64)
    input_dict = {
        'data_format': 'NHWC',
        'name': 'mp_grad_grad_3',
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: NHWC float32 with stride 2 and SAME padding
    orig_input = np.ones((1, 5, 5, 1), dtype=np.float32)
    orig_output = np.ones((1, 3, 3, 1), dtype=np.float32)
    grad = np.random.randn(1, 5, 5, 1).astype(np.float32)
    input_dict = {
        'data_format': 'NHWC',
        'name': 'mp_grad_grad_4',
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': [1, 3, 3, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: NHWC float32 with kernel size matching spatial dimensions
    orig_input = np.ones((1, 3, 3, 1), dtype=np.float32)
    orig_output = np.ones((1, 1, 1, 1), dtype=np.float32)
    grad = np.random.randn(1, 3, 3, 1).astype(np.float32)
    input_dict = {
        'data_format': 'NHWC',
        'name': 'mp_grad_grad_5',
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': [1, 3, 3, 1],
        'strides': [1, 1, 1, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: NHWC float32 with multiple channels
    orig_input = np.ones((1, 4, 4, 2), dtype=np.float32)
    orig_output = np.ones((1, 2, 2, 2), dtype=np.float32)
    grad = np.random.randn(1, 4, 4, 2).astype(np.float32)
    input_dict = {
        'data_format': 'NHWC',
        'name': 'mp_grad_grad_6',
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: NHWC float32 with batch size 2
    orig_input = np.ones((2, 6, 6, 1), dtype=np.float32)
    orig_output = np.ones((2, 3, 3, 1), dtype=np.float32)
    grad = np.random.randn(2, 6, 6, 1).astype(np.float32)
    input_dict = {
        'data_format': 'NHWC',
        'name': 'mp_grad_grad_7',
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: NHWC float64 with batch and multiple channels
    orig_input = np.ones((2, 2, 2, 2), dtype=np.float64)
    orig_output = np.ones((2, 1, 1, 2), dtype=np.float64)
    grad = np.random.randn(2, 2, 2, 2).astype(np.float64)
    input_dict = {
        'data_format': 'NHWC',
        'name': 'mp_grad_grad_8',
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'VALID'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: NHWC float32 with larger input size and SAME padding
    orig_input = np.ones((1, 8, 8, 1), dtype=np.float32)
    orig_output = np.ones((1, 4, 4, 1), dtype=np.float32)
    grad = np.random.randn(1, 8, 8, 1).astype(np.float32)
    input_dict = {
        'data_format': 'NHWC',
        'name': 'mp_grad_grad_9',
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: NHWC float32 with 8x8 input and multiple channels
    orig_input = np.ones((1, 8, 8, 3), dtype=np.float32)
    orig_output = np.ones((1, 4, 4, 3), dtype=np.float32)
    grad = np.random.randn(1, 8, 8, 3).astype(np.float32)
    input_dict = {
        'data_format': 'NHWC',
        'name': 'mp_grad_grad_10',
        'orig_input': orig_input,
        'orig_output': orig_output,
        'grad': grad,
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'SAME'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.MaxPoolGradGrad"] = tf_raw_ops_MaxPoolGradGrad_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MaxPoolGradGradV2_inputs():
    list_of_inputs = []
    
    # Case 1: Standard float32 NHWC
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'case_1',
        'orig_input': np.ones((1, 4, 4, 1), dtype=np.float32),
        'orig_output': np.ones((1, 2, 2, 1), dtype=np.float32),
        'grad': np.ones((1, 4, 4, 1), dtype=np.float32),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    # Case 2: float64 NHWC
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'case_2',
        'orig_input': np.zeros((1, 4, 4, 1), dtype=np.float64),
        'orig_output': np.zeros((1, 2, 2, 1), dtype=np.float64),
        'grad': np.zeros((1, 4, 4, 1), dtype=np.float64),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    # Case 3: SAME padding
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'case_3',
        'orig_input': np.ones((1, 4, 4, 1), dtype=np.float32),
        'orig_output': np.ones((1, 2, 2, 1), dtype=np.float32),
        'grad': np.ones((1, 4, 4, 1), dtype=np.float32),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'SAME'
    })

    # Case 4: Larger spatial dimensions NHWC
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'case_4',
        'orig_input': np.ones((1, 8, 8, 1), dtype=np.float32),
        'orig_output': np.ones((1, 4, 4, 1), dtype=np.float32),
        'grad': np.ones((1, 8, 8, 1), dtype=np.float32),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    # Case 5: Large shape, batch and channels NHWC
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'case_5',
        'orig_input': np.ones((2, 6, 6, 3), dtype=np.float32),
        'orig_output': np.ones((2, 2, 2, 3), dtype=np.float32),
        'grad': np.ones((2, 6, 6, 3), dtype=np.float32),
        'ksize': np.array([1, 3, 3, 1], dtype=np.int32),
        'strides': np.array([1, 3, 3, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    # Case 6: Different random values, float32 NHWC
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'case_6',
        'orig_input': np.random.rand(2, 6, 6, 3).astype(np.float32),
        'orig_output': np.random.rand(2, 2, 2, 3).astype(np.float32),
        'grad': np.random.rand(2, 6, 6, 3).astype(np.float32),
        'ksize': np.array([1, 3, 3, 1], dtype=np.int32),
        'strides': np.array([1, 3, 3, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    # Case 7: Non-symmetric strides and kernel sizes NHWC
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'case_7',
        'orig_input': np.ones((1, 5, 4, 1), dtype=np.float32),
        'orig_output': np.ones((1, 2, 3, 1), dtype=np.float32),
        'grad': np.ones((1, 5, 4, 1), dtype=np.float32),
        'ksize': np.array([1, 3, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 1, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    # Case 8: Negative values in inputs and grad NHWC
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'case_8',
        'orig_input': np.array([[[[-1.0], [2.0]], [[-3.0], [4.0]]]], dtype=np.float32),
        'orig_output': np.array([[[[4.0]]]], dtype=np.float32),
        'grad': np.array([[[[-0.5], [1.5]], [[-2.5], [3.5]]]], dtype=np.float32),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 1, 1, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    # Case 9: float64 type with multiple channels NHWC
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'case_9',
        'orig_input': np.ones((2, 4, 4, 2), dtype=np.float64),
        'orig_output': np.ones((2, 2, 2, 2), dtype=np.float64),
        'grad': np.ones((2, 4, 4, 2), dtype=np.float64),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    # Case 10: NHWC format with larger dimensions and 3 channels
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'case_10',
        'orig_input': np.ones((2, 8, 8, 3), dtype=np.float32),
        'orig_output': np.ones((2, 4, 4, 3), dtype=np.float32),
        'grad': np.ones((2, 8, 8, 3), dtype=np.float32),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.MaxPoolGradGradV2"] = tf_raw_ops_MaxPoolGradGradV2_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_MaxPoolGradV2_inputs():
    configs = [
        {
            "data_format": "NHWC", "padding": "VALID",
            "input_shape": [1, 2, 2, 1], "ksize": [1, 2, 2, 1], "strides": [1, 2, 2, 1]
        },
        {
            "data_format": "NHWC", "padding": "SAME",
            "input_shape": [1, 2, 2, 1], "ksize": [1, 2, 2, 1], "strides": [1, 1, 1, 1]
        },
        {
            "data_format": "NHWC", "padding": "VALID",
            "input_shape": [1, 3, 3, 1], "ksize": [1, 2, 2, 1], "strides": [1, 2, 2, 1]
        },
        {
            "data_format": "NHWC", "padding": "SAME",
            "input_shape": [1, 3, 3, 1], "ksize": [1, 2, 2, 1], "strides": [1, 1, 1, 1]
        },
        {
            "data_format": "NHWC", "padding": "SAME",
            "input_shape": [2, 3, 3, 2], "ksize": [1, 2, 2, 1], "strides": [1, 2, 2, 1]
        },
        {
            "data_format": "NHWC", "padding": "VALID",
            "input_shape": [2, 3, 3, 2], "ksize": [1, 2, 2, 1], "strides": [1, 1, 1, 1]
        },
        {
            "data_format": "NHWC", "padding": "SAME",
            "input_shape": [1, 4, 4, 1], "ksize": [1, 2, 2, 1], "strides": [1, 2, 2, 1]
        },
        {
            "data_format": "NHWC", "padding": "VALID",
            "input_shape": [1, 4, 4, 1], "ksize": [1, 2, 2, 1], "strides": [1, 1, 1, 1]
        },
        {
            "data_format": "NHWC", "padding": "VALID",
            "input_shape": [1, 2, 2, 3], "ksize": [1, 2, 2, 1], "strides": [1, 1, 1, 1]
        },
        {
            "data_format": "NHWC", "padding": "SAME",
            "input_shape": [2, 2, 2, 2], "ksize": [1, 1, 1, 1], "strides": [1, 1, 1, 1]
        }
    ]

    list_of_inputs = []
    
    with tf.device('/CPU:0'):
        for i, cfg in enumerate(configs):
            num_elements = int(np.prod(cfg["input_shape"]))
            arr = np.linspace(-2.0, 2.0, num_elements).astype(np.float32)
            orig_input = arr.reshape(cfg["input_shape"])
            
            ksize = cfg["ksize"]
            strides = cfg["strides"]
            padding = cfg["padding"]
            data_format = cfg["data_format"]
            
            orig_output = tf.raw_ops.MaxPool(
                input=orig_input,
                ksize=ksize,
                strides=strides,
                padding=padding,
                data_format="NHWC"
            ).numpy()
                
            orig_output = orig_output.astype(np.float32)
            grad_elements = int(np.prod(orig_output.shape))
            grad_arr = np.linspace(0.1, 1.0, grad_elements).astype(np.float32)
            grad = grad_arr.reshape(orig_output.shape)
            
            input_dict = {
                "padding": padding,
                "data_format": data_format,
                "name": f"maxpoolgradv2_{i}",
                "orig_input": orig_input,
                "orig_output": orig_output,
                "grad": grad,
                "ksize": np.array(ksize, dtype=np.int32),
                "strides": np.array(strides, dtype=np.int32)
            }
            list_of_inputs.append(copy.deepcopy(input_dict))
            
    return list_of_inputs

generated_inputs["tf.raw_ops.MaxPoolGradV2"] = tf_raw_ops_MaxPoolGradV2_inputs()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_MaxPoolV2_inputs():
    list_of_inputs = []

    # Input 1: Float32, NHWC, VALID
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'pool_1',
        'input': np.random.randn(2, 8, 8, 3).astype(np.float32),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    # Input 2: Float64, NHWC, SAME
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'pool_2',
        'input': np.random.randn(1, 10, 10, 1).astype(np.float64),
        'ksize': np.array([1, 3, 3, 1], dtype=np.int32),
        'strides': np.array([1, 1, 1, 1], dtype=np.int32),
        'padding': 'SAME'
    })

    # Input 3: Int8, NHWC, VALID
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'pool_3',
        'input': np.random.randint(-128, 127, size=(4, 8, 8, 3)).astype(np.int8),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    # Input 4: UInt8, NHWC, SAME
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'pool_4',
        'input': np.random.randint(0, 255, size=(1, 5, 5, 2)).astype(np.uint8),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 1, 1, 1], dtype=np.int32),
        'padding': 'SAME'
    })

    # Input 5: Int16, NHWC, VALID
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'pool_5',
        'input': np.random.randint(-32768, 32767, size=(2, 6, 6, 4)).astype(np.int16),
        'ksize': np.array([1, 3, 3, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    # Input 6: Float16, NHWC, SAME
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'pool_6',
        'input': np.random.randn(1, 4, 4, 1).astype(np.float16),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 1, 1, 1], dtype=np.int32),
        'padding': 'SAME'
    })

    # Input 7: Float32, NHWC, VALID
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'pool_7',
        'input': np.random.randn(2, 8, 8, 2).astype(np.float32),
        'ksize': np.array([1, 2, 2, 1], dtype=np.int32),
        'strides': np.array([1, 1, 1, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    # Input 8: Int64, NHWC, SAME
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'pool_8',
        'input': np.random.randint(-1000, 1000, size=(1, 12, 12, 2)).astype(np.int64),
        'ksize': np.array([1, 3, 3, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'SAME'
    })

    # Input 9: Float32, NHWC, VALID
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'pool_9',
        'input': np.random.randn(3, 16, 16, 3).astype(np.float32),
        'ksize': np.array([1, 4, 4, 1], dtype=np.int32),
        'strides': np.array([1, 4, 4, 1], dtype=np.int32),
        'padding': 'VALID'
    })

    # Input 10: Int32, NHWC, SAME
    list_of_inputs.append({
        'data_format': 'NHWC',
        'name': 'pool_10',
        'input': np.random.randint(-50, 50, size=(1, 14, 14, 4)).astype(np.int32),
        'ksize': np.array([1, 3, 3, 1], dtype=np.int32),
        'strides': np.array([1, 2, 2, 1], dtype=np.int32),
        'padding': 'SAME'
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.MaxPoolV2"] = tf_raw_ops_MaxPoolV2_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_extract_volume_patches_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "input": np.random.randn(1, 3, 3, 3, 1).astype(np.float32),
        "ksizes": [1, 2, 2, 2, 1],
        "strides": [1, 1, 1, 1, 1],
        "padding": "VALID",
        "name": "extract_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "input": np.random.randint(-10, 10, size=(2, 4, 4, 4, 3)).astype(np.int32),
        "ksizes": [1, 3, 3, 3, 1],
        "strides": [1, 2, 2, 2, 1],
        "padding": "SAME",
        "name": "extract_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "input": np.random.randint(0, 255, size=(1, 5, 5, 5, 2)).astype(np.uint8),
        "ksizes": [1, 1, 1, 1, 1],
        "strides": [1, 1, 1, 1, 1],
        "padding": "VALID",
        "name": "extract_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "input": np.random.randn(3, 2, 2, 2, 4).astype(np.float64),
        "ksizes": [1, 2, 2, 2, 1],
        "strides": [1, 1, 1, 1, 1],
        "padding": "SAME",
        "name": "extract_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "input": np.random.randn(1, 10, 10, 10, 1).astype(np.float32),
        "ksizes": [1, 5, 5, 5, 1],
        "strides": [1, 5, 5, 5, 1],
        "padding": "VALID",
        "name": "extract_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "input": np.random.randint(-1000, 1000, size=(2, 6, 6, 6, 2)).astype(np.int64),
        "ksizes": [1, 2, 3, 2, 1],
        "strides": [1, 1, 2, 1, 1],
        "padding": "SAME",
        "name": "extract_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "input": np.random.randn(1, 3, 4, 5, 1).astype(np.float32),
        "ksizes": [1, 2, 2, 2, 1],
        "strides": [1, 1, 1, 1, 1],
        "padding": "VALID",
        "name": "extract_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "input": np.random.randint(-50, 50, size=(4, 3, 3, 3, 2)).astype(np.int32),
        "ksizes": [1, 3, 2, 3, 1],
        "strides": [1, 1, 1, 1, 1],
        "padding": "SAME",
        "name": "extract_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "input": np.random.randint(0, 255, size=(1, 8, 8, 8, 3)).astype(np.uint8),
        "ksizes": [1, 4, 4, 4, 1],
        "strides": [1, 2, 2, 2, 1],
        "padding": "VALID",
        "name": "extract_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "input": np.random.randn(2, 2, 3, 4, 2).astype(np.float64),
        "ksizes": [1, 1, 2, 3, 1],
        "strides": [1, 1, 1, 1, 1],
        "padding": "SAME",
        "name": "extract_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.extract_volume_patches"] = tf_extract_volume_patches_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_nn_depthwise_conv2d_backprop_filter_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'input': np.random.randn(2, 5, 5, 3).astype(np.float32),
        'filter_sizes': np.array([3, 3, 3, 2], dtype=np.int32),
        'out_backprop': np.random.randn(2, 3, 3, 6).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'depthwise_conv_backprop_filter_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'input': np.random.randn(1, 6, 6, 2).astype(np.float32),
        'filter_sizes': np.array([2, 2, 2, 1], dtype=np.int32),
        'out_backprop': np.random.randn(1, 3, 3, 2).astype(np.float32),
        'strides': [1, 2, 2, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'depthwise_conv_backprop_filter_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'input': np.random.randn(3, 4, 4, 2).astype(np.float32),
        'filter_sizes': np.array([2, 2, 2, 3], dtype=np.int32),
        'out_backprop': np.random.randn(3, 3, 3, 6).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'depthwise_conv_backprop_filter_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'input': np.random.randn(2, 8, 8, 2).astype(np.float32),
        'filter_sizes': np.array([3, 3, 2, 1], dtype=np.int32),
        'out_backprop': np.random.randn(2, 8, 8, 2).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'depthwise_conv_backprop_filter_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'input': np.random.randn(1, 3, 3, 1).astype(np.float32),
        'filter_sizes': np.array([2, 2, 1, 1], dtype=np.int32),
        'out_backprop': np.random.randn(1, 2, 2, 1).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'depthwise_conv_backprop_filter_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'input': np.random.randn(1, 7, 7, 1).astype(np.float32),
        'filter_sizes': np.array([3, 3, 1, 2], dtype=np.int32),
        'out_backprop': np.random.randn(1, 4, 4, 2).astype(np.float32),
        'strides': [1, 2, 2, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'depthwise_conv_backprop_filter_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'input': np.random.randn(2, 5, 5, 2).astype(np.float32),
        'filter_sizes': np.array([3, 3, 2, 2], dtype=np.int32),
        'out_backprop': np.random.randn(2, 2, 2, 4).astype(np.float32),
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'depthwise_conv_backprop_filter_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'input': np.random.randn(1, 10, 10, 2).astype(np.float32),
        'filter_sizes': np.array([3, 3, 2, 1], dtype=np.int32),
        'out_backprop': np.random.randn(1, 4, 4, 2).astype(np.float32),
        'strides': [1, 2, 2, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'depthwise_conv_backprop_filter_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'input': -np.random.rand(1, 4, 4, 1).astype(np.float32),
        'filter_sizes': np.array([2, 2, 1, 1], dtype=np.int32),
        'out_backprop': -np.random.rand(1, 4, 4, 1).astype(np.float32),
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'depthwise_conv_backprop_filter_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'input': np.random.randn(1, 10, 10, 3).astype(np.float32),
        'filter_sizes': np.array([4, 4, 3, 1], dtype=np.int32),
        'out_backprop': np.random.randn(1, 3, 3, 3).astype(np.float32),
        'strides': [1, 3, 3, 1],
        'padding': 'VALID',
        'data_format': 'NHWC',
        'dilations': [1, 1, 1, 1],
        'name': 'depthwise_conv_backprop_filter_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.depthwise_conv2d_backprop_filter"] = tf_nn_depthwise_conv2d_backprop_filter_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_strings_unicode_encode_inputs():
    list_of_inputs = []

    # Input 1: Standard UTF-8, 2D array, valid ASCII characters
    input_val = np.array([[71, 111, 111, 100], [109, 111, 114, 110]], dtype=np.int32)
    input_dict = {
        "input": input_val,
        "output_encoding": "UTF-8",
        "errors": "replace",
        "replacement_char": 65533,
        "name": "encode_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: UTF-16-BE, 1D array
    input_val = np.array([71, 111, 111, 100], dtype=np.int32)
    input_dict = {
        "input": input_val,
        "output_encoding": "UTF-16-BE",
        "errors": "replace",
        "replacement_char": 65533,
        "name": "encode_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: UTF-32-BE, 3D array
    input_val = np.array([[[65, 66], [67, 68]], [[69, 70], [71, 72]]], dtype=np.int32)
    input_dict = {
        "input": input_val,
        "output_encoding": "UTF-32-BE",
        "errors": "replace",
        "replacement_char": 65533,
        "name": "encode_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Errors "ignore" with invalid surrogate codepoint
    input_val = np.array([[65, 0xD800, 66]], dtype=np.int32)
    input_dict = {
        "input": input_val,
        "output_encoding": "UTF-8",
        "errors": "ignore",
        "replacement_char": 65533,
        "name": "encode_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Errors "strict", with emoji characters
    input_val = np.array([[128522, 128523]], dtype=np.int32)
    input_dict = {
        "input": input_val,
        "output_encoding": "UTF-8",
        "errors": "strict",
        "replacement_char": 65533,
        "name": "encode_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Custom replacement character for invalid codepoint
    input_val = np.array([[65, 0x110000, 66]], dtype=np.int32)
    input_dict = {
        "input": input_val,
        "output_encoding": "UTF-8",
        "errors": "replace",
        "replacement_char": 63,
        "name": "encode_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Int32 input type, UTF-16-BE encoding
    input_val = np.array([[104, 101, 108, 108, 111]], dtype=np.int32)
    input_dict = {
        "input": input_val,
        "output_encoding": "UTF-16-BE",
        "errors": "strict",
        "replacement_char": 65533,
        "name": "encode_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty last dimension
    input_val = np.empty((2, 0), dtype=np.int32)
    input_dict = {
        "input": input_val,
        "output_encoding": "UTF-8",
        "errors": "replace",
        "replacement_char": 65533,
        "name": "encode_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Emojis with UTF-32-BE encoding
    input_val = np.array([[0x1F600, 0x1F601, 0x1F602]], dtype=np.int32)
    input_dict = {
        "input": input_val,
        "output_encoding": "UTF-32-BE",
        "errors": "strict",
        "replacement_char": 65533,
        "name": "encode_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single character elements along the last axis
    input_val = np.array([[65], [66], [67]], dtype=np.int32)
    input_dict = {
        "input": input_val,
        "output_encoding": "UTF-8",
        "errors": "replace",
        "replacement_char": 65533,
        "name": "encode_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.strings.unicode_encode"] = tf_strings_unicode_encode_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_audio_encode_wav_inputs():
    list_of_inputs = []

    # Input 1: Standard stereo, 44100 Hz
    audio = np.random.uniform(-1.0, 1.0, size=(16000, 2)).astype(np.float32)
    sample_rate = np.int32(44100)
    name = "stereo_44k"
    list_of_inputs.append({"audio": audio, "sample_rate": sample_rate, "name": name})

    # Input 2: Mono, 16000 Hz
    audio = np.random.uniform(-1.0, 1.0, size=(8000, 1)).astype(np.float32)
    sample_rate = np.int32(16000)
    name = "mono_16k"
    list_of_inputs.append({"audio": audio, "sample_rate": sample_rate, "name": name})

    # Input 3: Stereo, 22050 Hz, clamping test (values out of range)
    audio = np.random.uniform(-2.0, 2.0, size=(11025, 2)).astype(np.float32)
    sample_rate = np.int32(22050)
    name = "clamped_stereo"
    list_of_inputs.append({"audio": audio, "sample_rate": sample_rate, "name": name})

    # Input 4: 5.1 channel (6 channels), 48000 Hz
    audio = np.random.uniform(-0.5, 0.5, size=(24000, 6)).astype(np.float32)
    sample_rate = np.int32(48000)
    name = "surround_sound"
    list_of_inputs.append({"audio": audio, "sample_rate": sample_rate, "name": name})

    # Input 5: Short audio, mono, 8000 Hz
    audio = np.random.uniform(-1.0, 1.0, size=(100, 1)).astype(np.float32)
    sample_rate = np.int32(8000)
    name = "short_beep"
    list_of_inputs.append({"audio": audio, "sample_rate": sample_rate, "name": name})

    # Input 6: Large input length
    audio = np.random.uniform(-1.0, 1.0, size=(100000, 2)).astype(np.float32)
    sample_rate = np.int32(44100)
    name = "long_audio"
    list_of_inputs.append({"audio": audio, "sample_rate": sample_rate, "name": name})

    # Input 7: Empty length (0, 2)
    audio = np.empty((0, 2), dtype=np.float32)
    sample_rate = np.int32(44100)
    name = "empty"
    list_of_inputs.append({"audio": audio, "sample_rate": sample_rate, "name": name})

    # Input 8: All negative float values, stereo, 44100 Hz
    audio = np.random.uniform(-1.0, 0.0, size=(8000, 2)).astype(np.float32)
    sample_rate = np.int32(44100)
    name = "negative_values"
    list_of_inputs.append({"audio": audio, "sample_rate": sample_rate, "name": name})

    # Input 9: Small sample rate, 4000 Hz
    audio = np.random.uniform(-0.8, 0.8, size=(4000, 1)).astype(np.float32)
    sample_rate = np.int32(4000)
    name = "low_rate"
    list_of_inputs.append({"audio": audio, "sample_rate": sample_rate, "name": name})

    # Input 10: High sample rate, 96000 Hz
    audio = np.random.uniform(-1.0, 1.0, size=(96000, 2)).astype(np.float32)
    sample_rate = np.int32(96000)
    name = "high_rate"
    list_of_inputs.append({"audio": audio, "sample_rate": sample_rate, "name": name})

    return list_of_inputs

generated_inputs["tf.audio.encode_wav"] = tf_audio_encode_wav_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_batch_to_space_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D spatial dimensions, float32, no cropping
    input_dict = {
        "input": np.arange(4, dtype=np.float32).reshape(4, 1, 1, 1),
        "block_shape": np.array([2, 2], dtype=np.int32),
        "crops": np.array([[0, 0], [0, 0]], dtype=np.int32),
        "name": "op1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D spatial, float32 with multiple channels, no cropping
    input_dict = {
        "input": np.arange(12, dtype=np.float32).reshape(4, 1, 1, 3),
        "block_shape": np.array([2, 2], dtype=np.int32),
        "crops": np.array([[0, 0], [0, 0]], dtype=np.int32),
        "name": "op2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D spatial, int32 input, int64 block_shape and crops
    input_dict = {
        "input": np.arange(16, dtype=np.int32).reshape(4, 2, 2, 1),
        "block_shape": np.array([2, 2], dtype=np.int64),
        "crops": np.array([[0, 0], [0, 0]], dtype=np.int64),
        "name": "op3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D spatial, float64, with cropping on one dimension
    input_dict = {
        "input": np.arange(24, dtype=np.float64).reshape(8, 1, 3, 1),
        "block_shape": np.array([2, 2], dtype=np.int32),
        "crops": np.array([[0, 0], [2, 0]], dtype=np.int32),
        "name": "op4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D spatial dimension, cropping on both sides
    input_dict = {
        "input": np.arange(24, dtype=np.float32).reshape(6, 2, 2),
        "block_shape": np.array([3], dtype=np.int32),
        "crops": np.array([[1, 1]], dtype=np.int32),
        "name": "op5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D spatial dimensions, complex block shape and crops
    input_dict = {
        "input": np.arange(32, dtype=np.float32).reshape(4, 2, 2, 2, 1),
        "block_shape": np.array([2, 1, 2], dtype=np.int32),
        "crops": np.array([[0, 1], [0, 0], [1, 1]], dtype=np.int32),
        "name": "op6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative values in the input tensor
    input_dict = {
        "input": np.arange(-16, 16, dtype=np.float32).reshape(4, 2, 2, 2),
        "block_shape": np.array([2, 2], dtype=np.int32),
        "crops": np.array([[1, 0], [0, 1]], dtype=np.int32),
        "name": "op7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Random float32 values, 1D spatial dimension
    input_dict = {
        "input": np.random.normal(size=(4, 1, 5)).astype(np.float32),
        "block_shape": np.array([4], dtype=np.int32),
        "crops": np.array([[0, 3]], dtype=np.int32),
        "name": "op8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Random int32 values, non-symmetric block shape and crops
    input_dict = {
        "input": np.random.randint(-100, 100, size=(12, 1, 1, 2)).astype(np.int32),
        "block_shape": np.array([3, 2], dtype=np.int64),
        "crops": np.array([[0, 2], [1, 1]], dtype=np.int64),
        "name": "op9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D spatial dimensions, block shape elements are all 1
    input_dict = {
        "input": np.arange(16, dtype=np.float32).reshape(1, 2, 2, 2, 2, 1),
        "block_shape": np.array([1, 1, 1, 1], dtype=np.int32),
        "crops": np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32),
        "name": "op10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.batch_to_space"] = tf_batch_to_space_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_boolean_mask_inputs():
    list_of_inputs = []
    
    # Input 1
    input_dict = {
        "tensor": np.array([10, 20, 30], dtype=np.int64),
        "mask": np.array([True, False, True], dtype=bool),
        "axis": np.array(0, dtype=np.int32),
        "name": "case1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input_dict = {
        "tensor": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "mask": np.array([False, True], dtype=bool),
        "axis": np.array(0, dtype=np.int32),
        "name": "case2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input_dict = {
        "tensor": np.array([[[1], [2]], [[3], [4]]], dtype=np.int32),
        "mask": np.array([True, True], dtype=bool),
        "axis": np.array(0, dtype=np.int32),
        "name": "case3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_dict = {
        "tensor": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "mask": np.array([[True, False], [True, True]], dtype=bool),
        "axis": np.array(0, dtype=np.int32),
        "name": "case4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_dict = {
        "tensor": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32),
        "mask": np.array([[True, False], [False, True]], dtype=bool),
        "axis": np.array(0, dtype=np.int32),
        "name": "case5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_dict = {
        "tensor": np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32),
        "mask": np.array([True, False, True], dtype=bool),
        "axis": np.array(1, dtype=np.int32),
        "name": "case6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_dict = {
        "tensor": np.array([1.5, -2.5, 3.5], dtype=np.float64),
        "mask": np.array([True, True, False], dtype=bool),
        "axis": np.array(0, dtype=np.int32),
        "name": "case7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_dict = {
        "tensor": np.array([[[1.1, 1.2], [1.3, 1.4]], [[2.1, 2.2], [2.3, 2.4]]], dtype=np.float32),
        "mask": np.array([True, False], dtype=bool),
        "axis": np.array(1, dtype=np.int32),
        "name": "case8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_dict = {
        "tensor": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        "mask": np.array([True, True, False, False], dtype=bool),
        "axis": np.array(0, dtype=np.int32),
        "name": "case9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_dict = {
        "tensor": np.arange(24).reshape(2, 3, 4).astype(np.int32),
        "mask": np.array([True, False, True], dtype=bool),
        "axis": np.array(1, dtype=np.int32),
        "name": "case10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.boolean_mask"] = tf_boolean_mask_inputs()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_dlpack_to_dlpack_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 tensor
    tf_tensor = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
    list_of_inputs.append(copy.deepcopy({"tf_tensor": tf_tensor}))

    # Input 2: 2D int32 tensor with negative values
    tf_tensor = tf.constant([[-1, 2], [3, -4]], dtype=tf.int32)
    list_of_inputs.append(copy.deepcopy({"tf_tensor": tf_tensor}))

    # Input 3: 3D float64 tensor
    tf_tensor = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=tf.float64)
    list_of_inputs.append(copy.deepcopy({"tf_tensor": tf_tensor}))

    # Input 4: 0D (scalar) float32 tensor
    tf_tensor = tf.constant(42.0, dtype=tf.float32)
    list_of_inputs.append(copy.deepcopy({"tf_tensor": tf_tensor}))

    # Input 5: 4D int64 tensor
    tf_tensor = tf.constant([[[[1, 2]], [[3, 4]]]], dtype=tf.int64)
    list_of_inputs.append(copy.deepcopy({"tf_tensor": tf_tensor}))

    # Input 6: 1D float16 tensor
    tf_tensor = tf.constant([0.1, -0.2, 0.3], dtype=tf.float16)
    list_of_inputs.append(copy.deepcopy({"tf_tensor": tf_tensor}))

    # Input 7: 2D uint8 tensor
    tf_tensor = tf.constant([[0, 255], [128, 64]], dtype=tf.uint8)
    list_of_inputs.append(copy.deepcopy({"tf_tensor": tf_tensor}))

    # Input 8: 3D int8 tensor
    tf_tensor = tf.constant([[[1, -1]], [[2, -2]]], dtype=tf.int8)
    list_of_inputs.append(copy.deepcopy({"tf_tensor": tf_tensor}))

    # Input 9: Large 1D float32 tensor
    tf_tensor = tf.random.uniform(shape=[100], minval=-1.0, maxval=1.0, dtype=tf.float32)
    list_of_inputs.append(copy.deepcopy({"tf_tensor": tf_tensor}))

    # Input 10: 5D float32 tensor
    tf_tensor = tf.zeros(shape=(2, 2, 2, 2, 2), dtype=tf.float32)
    list_of_inputs.append(copy.deepcopy({"tf_tensor": tf_tensor}))

    return list_of_inputs

generated_inputs["tf.experimental.dlpack.to_dlpack"] = tf_experimental_dlpack_to_dlpack_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_crop_inputs():
    list_of_inputs = []

    # Input 1: 3D float32 image
    input_dict = {
        "value": np.random.randn(10, 10, 3).astype(np.float32),
        "size": np.array([5, 5, 3], dtype=np.int32),
        "seed": np.array([1, 2], dtype=np.int32),
        "name": "crop_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D uint8 image (e.g. MNIST size)
    input_dict = {
        "value": np.random.randint(0, 256, size=(28, 28, 1)).astype(np.uint8),
        "size": np.array([14, 14, 1], dtype=np.int32),
        "seed": np.array([42, 43], dtype=np.int32),
        "name": "crop_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4D batch of float64 images
    input_dict = {
        "value": np.random.randn(4, 32, 32, 3).astype(np.float64),
        "size": np.array([2, 16, 16, 3], dtype=np.int32),
        "seed": np.array([10, 20], dtype=np.int32),
        "name": "crop_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D integer array
    input_dict = {
        "value": np.random.randint(-100, 100, size=(100, 100)).astype(np.int32),
        "size": np.array([50, 50], dtype=np.int32),
        "seed": np.array([100, 200], dtype=np.int32),
        "name": "crop_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D float32 vector
    input_dict = {
        "value": np.random.randn(10).astype(np.float32),
        "size": np.array([5], dtype=np.int32),
        "seed": np.array([0, 0], dtype=np.int32),
        "name": "crop_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 5D video-like tensor
    input_dict = {
        "value": np.random.randn(2, 5, 16, 16, 3).astype(np.float32),
        "size": np.array([1, 3, 8, 8, 3], dtype=np.int32),
        "seed": np.array([9, 9], dtype=np.int32),
        "name": "crop_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Crop same size as input (identity crop)
    input_dict = {
        "value": np.random.randint(0, 100, size=(8, 8, 3)).astype(np.int64),
        "size": np.array([8, 8, 3], dtype=np.int32),
        "seed": np.array([7, 8], dtype=np.int32),
        "name": "crop_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Crop with size 1 along each dimension
    input_dict = {
        "value": np.random.randn(5, 5, 5).astype(np.float32),
        "size": np.array([1, 1, 1], dtype=np.int32),
        "seed": np.array([123, 456], dtype=np.int32),
        "name": "crop_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large values, using int32 size and seed
    input_dict = {
        "value": np.random.randn(12, 12, 4).astype(np.float64),
        "size": np.array([6, 6, 2], dtype=np.int32),
        "seed": np.array([55, 66], dtype=np.int32),
        "name": "crop_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Int16 high resolution image
    input_dict = {
        "value": np.random.randint(-32768, 32767, size=(256, 256, 3)).astype(np.int16),
        "size": np.array([128, 128, 3], dtype=np.int32),
        "seed": np.array([1111, 2222], dtype=np.int32),
        "name": "crop_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.image.stateless_random_crop"] = tf_image_stateless_random_crop_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_io_encode_png_inputs():
    list_of_inputs = []
    
    # Input 1: Grayscale (1 channel), uint8, default compression
    input_dict = {
        'image': np.zeros((10, 10, 1), dtype=np.uint8),
        'compression': -1,
        'name': "gray_img"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Grayscale + Alpha (2 channels), uint8, compression 0
    input_dict = {
        'image': np.ones((8, 8, 2), dtype=np.uint8) * 128,
        'compression': 0,
        'name': "gray_alpha_img"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: RGB (3 channels), uint8, max compression
    input_dict = {
        'image': np.random.randint(0, 256, size=(12, 12, 3), dtype=np.uint8),
        'compression': 9,
        'name': "rgb_img"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: RGBA (4 channels), uint8, compression 5
    input_dict = {
        'image': np.zeros((16, 16, 4), dtype=np.uint8),
        'compression': 5,
        'name': "rgba_img"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batch of Grayscale, uint8, compression 1
    input_dict = {
        'image': np.ones((2, 10, 10, 1), dtype=np.uint8) * 200,
        'compression': 1,
        'name': "batch_gray"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batch of RGB, uint8, default compression
    input_dict = {
        'image': np.zeros((1, 15, 15, 3), dtype=np.uint8),
        'compression': -1,
        'name': "batch_rgb"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: RGB, uint8, compression 6
    input_dict = {
        'image': np.random.randint(0, 256, size=(20, 20, 3), dtype=np.uint8),
        'compression': 6,
        'name': "rgb_uint8_alternative"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Grayscale, uint8, compression 3
    input_dict = {
        'image': np.ones((5, 5, 1), dtype=np.uint8) * 128,
        'compression': 3,
        'name': "gray_uint8_alternative"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: RGBA, uint8, compression 8
    input_dict = {
        'image': np.zeros((2, 2, 4), dtype=np.uint8),
        'compression': 8,
        'name': "rgba_uint8_alternative"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Rank 5 Batch of RGB, uint8, compression 2
    input_dict = {
        'image': np.ones((1, 2, 4, 4, 3), dtype=np.uint8) * 100,
        'compression': 2,
        'name': "rank5_rgb"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.io.encode_png"] = tf_io_encode_png_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_io_serialize_tensor_inputs():
    list_of_inputs = []

    # Input 1: 0D int32 scalar
    input_dict = {
        "tensor": np.array(42, dtype=np.int32),
        "name": "scalar_int"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float32 array with negative values
    input_dict = {
        "tensor": np.array([-1.2, 0.0, 3.14], dtype=np.float32),
        "name": "float_1d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D boolean array
    input_dict = {
        "tensor": np.array([[True, False], [False, True]], dtype=bool),
        "name": "bool_2d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D int64 array
    input_dict = {
        "tensor": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64),
        "name": "int64_3d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D float64 array with special values
    input_dict = {
        "tensor": np.array([np.inf, -np.inf, np.nan], dtype=np.float64),
        "name": "float64_specials"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D uint8 array
    input_dict = {
        "tensor": np.array([[0, 255], [128, 64]], dtype=np.uint8),
        "name": "uint8_2d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 array
    input_dict = {
        "tensor": np.random.uniform(-1, 1, (2, 3, 3, 3)).astype(np.float32),
        "name": "float_4d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D int32 array (containing zeros and ones)
    input_dict = {
        "tensor": np.ones((2, 2, 2), dtype=np.int32),
        "name": "int32_3d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty 1D float32 array
    input_dict = {
        "tensor": np.array([], dtype=np.float32),
        "name": "empty_float"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D int64 array with large values
    input_dict = {
        "tensor": np.array([9223372036854775807, -9223372036854775808, 0], dtype=np.int64),
        "name": "int64_large"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.io.serialize_tensor"] = tf_io_serialize_tensor_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_linalg_inv_inputs():
    list_of_inputs = []

    # Input 1: 2D float32
    input_val = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append({
        "input": input_val,
        "adjoint": False,
        "name": "inv_1"
    })

    # Input 2: 2D float64 with adjoint=True
    input_val = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.float64)
    list_of_inputs.append({
        "input": input_val,
        "adjoint": True,
        "name": "inv_2"
    })

    # Input 3: complex64
    input_val = np.array([[1.0 + 1.0j, 0.0], [0.0, 2.0 - 1.0j]], dtype=np.complex64)
    list_of_inputs.append({
        "input": input_val,
        "adjoint": False,
        "name": "inv_3"
    })

    # Input 4: complex128 with adjoint=True
    input_val = np.array([[2.0j, 1.0], [1.0, -2.0j]], dtype=np.complex128)
    list_of_inputs.append({
        "input": input_val,
        "adjoint": True,
        "name": "inv_4"
    })

    # Input 5: float16 (half)
    input_val = np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=np.float16)
    list_of_inputs.append({
        "input": input_val,
        "adjoint": False,
        "name": "inv_5"
    })

    # Input 6: 3D float32 (batch of 2x2 matrices)
    input_val = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    list_of_inputs.append({
        "input": input_val,
        "adjoint": False,
        "name": "inv_6"
    })

    # Input 7: 4D float64 (batch of 3x2x2)
    input_val = np.array([
        [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]],
        [[[3.0, 0.0], [0.0, 3.0]], [[4.0, 0.0], [0.0, 4.0]]],
        [[[-1.0, 0.0], [0.0, -1.0]], [[-2.0, 0.0], [0.0, -2.0]]]
    ], dtype=np.float64)
    list_of_inputs.append({
        "input": input_val,
        "adjoint": True,
        "name": "inv_7"
    })

    # Input 8: 2D float32 with negative values
    input_val = np.array([[-1.0, -2.0], [-3.0, -5.0]], dtype=np.float32)
    list_of_inputs.append({
        "input": input_val,
        "adjoint": False,
        "name": "inv_8"
    })

    # Input 9: 2D float32 identity (5x5)
    input_val = np.eye(5, dtype=np.float32)
    list_of_inputs.append({
        "input": input_val,
        "adjoint": False,
        "name": "inv_9"
    })

    # Input 10: 3D complex64 batch
    input_val = np.array([
        [[1.0j, 0.0], [0.0, 1.0j]],
        [[2.0j, 0.0], [0.0, 2.0j]]
    ], dtype=np.complex64)
    list_of_inputs.append({
        "input": input_val,
        "adjoint": True,
        "name": "inv_10"
    })

    return list_of_inputs

generated_inputs["tf.linalg.inv"] = tf_linalg_inv_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_tridiagonal_matmul_inputs():
    list_of_inputs = []

    # Input 1: Basic float32, no batch
    diagonals = np.array([
        [1.0, 2.0, 0.0],  # superdiag
        [3.0, 4.0, 5.0],  # maindiag
        [0.0, 6.0, 7.0]   # subdiag
    ], dtype=np.float32)
    rhs = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0]
    ], dtype=np.float32)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'name': 'matmul_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, with negative values
    diagonals = np.array([
        [-1.0, -2.0, 0.0, 0.0],
        [2.0, 2.0, -2.0, 2.0],
        [0.0, -1.0, -1.0, -3.0]
    ], dtype=np.float64)
    rhs = np.array([
        [1.0],
        [-1.0],
        [2.0],
        [-2.0]
    ], dtype=np.float64)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'name': 'matmul_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Batch dimension 1, float32
    diagonals = np.random.randn(2, 3, 5).astype(np.float32)
    rhs = np.random.randn(2, 5, 3).astype(np.float32)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'name': 'matmul_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex64
    diagonals = (np.random.randn(3, 4) + 1j * np.random.randn(3, 4)).astype(np.complex64)
    rhs = (np.random.randn(4, 2) + 1j * np.random.randn(4, 2)).astype(np.complex64)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'name': 'matmul_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex128, batched
    diagonals = (np.random.randn(2, 3, 3) + 1j * np.random.randn(2, 3, 3)).astype(np.complex128)
    rhs = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(np.complex128)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'name': 'matmul_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large M and N, float64
    diagonals = np.random.randn(3, 100).astype(np.float64)
    rhs = np.random.randn(100, 50).astype(np.float64)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'name': 'matmul_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Batch dimensions 2, float32
    diagonals = np.random.randn(3, 2, 3, 6).astype(np.float32)
    rhs = np.random.randn(3, 2, 6, 2).astype(np.float32)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'name': 'matmul_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small size M=2, N=1
    diagonals = np.array([
        [0.5, 0.0],
        [1.5, 2.5],
        [0.0, 3.5]
    ], dtype=np.float32)
    rhs = np.array([
        [1.0],
        [2.0]
    ], dtype=np.float32)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'name': 'matmul_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Zero arrays
    diagonals = np.zeros((3, 10), dtype=np.float32)
    rhs = np.zeros((10, 10), dtype=np.float32)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'name': 'matmul_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single batch large size, float32
    diagonals = np.random.randn(5, 3, 20).astype(np.float32)
    rhs = np.random.randn(5, 20, 10).astype(np.float32)
    input_dict = {
        'diagonals': diagonals,
        'rhs': rhs,
        'diagonals_format': 'compact',
        'name': 'matmul_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.tridiagonal_matmul"] = tf_linalg_tridiagonal_matmul_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_math_argmin_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis 0
    input_dict = {
        "input": np.array([1.0, 10.0, 26.9, 2.8, 166.32, 62.3], dtype=np.float32),
        "axis": np.array(0, dtype=np.int32),
        "output_type": np.int64,
        "name": "argmin_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 array, axis 1
    input_dict = {
        "input": np.array([[2.0, 3.0, 1.0], [5.0, -1.0, 4.0]], dtype=np.float64),
        "axis": np.array(1, dtype=np.int64),
        "output_type": np.int32,
        "name": "argmin_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D int32 array with negative values, axis -1
    input_dict = {
        "input": np.array([[[-1, -2], [3, 4]], [[5, -6], [7, 8]]], dtype=np.int32),
        "axis": np.array(-1, dtype=np.int32),
        "output_type": np.int64,
        "name": "argmin_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D float32 array, axis 0
    input_dict = {
        "input": np.array([10.0, 20.0, 5.0, 40.0], dtype=np.float32),
        "axis": np.array(0, dtype=np.int32),
        "output_type": np.int32,
        "name": "argmin_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D random float32 array, axis 2
    input_dict = {
        "input": np.random.randn(2, 3, 4, 5).astype(np.float32),
        "axis": np.array(2, dtype=np.int64),
        "output_type": np.int64,
        "name": "argmin_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D int32 array, axis -2
    input_dict = {
        "input": np.array([[10, 20], [30, 40]], dtype=np.int32),
        "axis": np.array(-2, dtype=np.int32),
        "output_type": np.int32,
        "name": "argmin_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float64 array with duplicates to test smallest index tie breaker
    input_dict = {
        "input": np.array([5.0, 2.0, 2.0, 8.0], dtype=np.float64),
        "axis": np.array(0, dtype=np.int64),
        "output_type": np.int64,
        "name": "argmin_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D int32 array, axis 1
    input_dict = {
        "input": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32),
        "axis": np.array(1, dtype=np.int32),
        "output_type": np.int32,
        "name": "argmin_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32 array, axis 0
    input_dict = {
        "input": np.array([[1.5, 2.5], [0.5, 3.5]], dtype=np.float32),
        "axis": np.array(0, dtype=np.int32),
        "output_type": np.int64,
        "name": "argmin_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D int64 array, axis 4
    input_dict = {
        "input": np.ones((2, 2, 2, 2, 2), dtype=np.int64),
        "axis": np.array(4, dtype=np.int64),
        "output_type": np.int32,
        "name": "argmin_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.argmin"] = tf_math_argmin_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_scalar_mul_inputs():
    list_of_inputs = []

    # Input 1: Float32 scalar and 1D float32 array
    list_of_inputs.append({
        "scalar": np.array(2.5, dtype=np.float32),
        "x": np.array([1.0, -2.0, 3.0], dtype=np.float32),
        "name": "scale_1d_float"
    })

    # Input 2: Int32 negative scalar and 2D int32 array
    list_of_inputs.append({
        "scalar": np.array(-5, dtype=np.int32),
        "x": np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.int32),
        "name": "scale_2d_int"
    })

    # Input 3: Float64 zero scalar and 3D float64 array
    list_of_inputs.append({
        "scalar": np.array(0.0, dtype=np.float64),
        "x": np.arange(24, dtype=np.float64).reshape((2, 3, 4)),
        "name": "scale_3d_zero"
    })

    # Input 4: Complex scalar and 1D complex array
    list_of_inputs.append({
        "scalar": np.array(1.5 + 2.0j, dtype=np.complex64),
        "x": np.array([1.0 - 1.0j, 2.0 + 3.0j], dtype=np.complex64),
        "name": "scale_complex"
    })

    # Input 5: Float32 scalar and 0D float32 scalar tensor
    list_of_inputs.append({
        "scalar": np.array(-1.2, dtype=np.float32),
        "x": np.array(4.5, dtype=np.float32),
        "name": "scale_0d"
    })

    # Input 6: Large float32 scalar and 4D float32 array
    list_of_inputs.append({
        "scalar": np.array(100.0, dtype=np.float32),
        "x": np.ones((2, 2, 2, 2), dtype=np.float32),
        "name": "scale_4d_large"
    })

    # Input 7: Float16 scalar and 2D float16 array
    list_of_inputs.append({
        "scalar": np.array(0.5, dtype=np.float16),
        "x": np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float16),
        "name": "scale_float16"
    })

    # Input 8: Int64 scalar and 1D int64 array
    list_of_inputs.append({
        "scalar": np.array(10, dtype=np.int64),
        "x": np.array([100, 200, 300], dtype=np.int64),
        "name": "scale_int64"
    })

    # Input 9: Small float64 scalar and 5D float64 array
    list_of_inputs.append({
        "scalar": np.array(1e-5, dtype=np.float64),
        "x": np.ones((1, 2, 1, 2, 1), dtype=np.float64),
        "name": "scale_5d_small"
    })

    # Input 10: Negative float32 scalar and large 3D float32 array
    list_of_inputs.append({
        "scalar": np.array(-0.1, dtype=np.float32),
        "x": np.random.randn(5, 5, 5).astype(np.float32),
        "name": "scale_random_3d"
    })

    return list_of_inputs

generated_inputs["tf.math.scalar_mul"] = tf_math_scalar_mul_inputs()

import numpy as np
import copy
import tensorflow as tf

def tf_nn_ctc_loss_inputs():
    list_of_inputs = []

    # Input 1
    labels = np.array([[1, 2, 0], [2, 1, 1]], dtype=np.int32)
    logits = np.random.uniform(size=(5, 2, 4)).astype(np.float32)
    label_length = np.array([2, 3], dtype=np.int32)
    logit_length = np.array([5, 5], dtype=np.int32)
    logits_time_major = True
    unique = np.array([], dtype=np.int32)
    blank_index = 0
    name = "ctc_loss_1"
    list_of_inputs.append({
        'labels': labels,
        'logits': logits,
        'label_length': label_length,
        'logit_length': logit_length,
        'logits_time_major': logits_time_major,
        'unique': unique,
        'blank_index': blank_index,
        'name': name
    })

    # Input 2
    labels = np.array([[1, 2, 3, 0, 0], [4, 5, 1, 2, 0], [3, 2, 1, 4, 5], [1, 0, 0, 0, 0]], dtype=np.int64)
    logits = np.random.uniform(size=(4, 10, 8)).astype(np.float32)
    label_length = np.array([3, 4, 5, 1], dtype=np.int64)
    logit_length = np.array([10, 10, 10, 10], dtype=np.int64)
    logits_time_major = False
    unique = np.array([], dtype=np.int64)
    blank_index = -1
    name = "ctc_loss_2"
    list_of_inputs.append({
        'labels': labels,
        'logits': logits,
        'label_length': label_length,
        'logit_length': logit_length,
        'logits_time_major': logits_time_major,
        'unique': unique,
        'blank_index': blank_index,
        'name': name
    })

    # Input 3
    labels = np.array([[1, 2]], dtype=np.int32)
    logits = np.random.uniform(size=(4, 1, 3)).astype(np.float32)
    label_length = np.array([2], dtype=np.int32)
    logit_length = np.array([4], dtype=np.int32)
    logits_time_major = True
    unique = np.array([], dtype=np.int32)
    blank_index = 0
    name = "ctc_loss_3"
    list_of_inputs.append({
        'labels': labels,
        'logits': logits,
        'label_length': label_length,
        'logit_length': logit_length,
        'logits_time_major': logits_time_major,
        'unique': unique,
        'blank_index': blank_index,
        'name': name
    })

    # Input 4
    labels = np.array([[1, 2, 3, 4], [4, 3, 2, 1], [2, 3, 1, 0]], dtype=np.int32)
    logits = np.random.uniform(size=(3, 15, 6)).astype(np.float32)
    label_length = np.array([4, 4, 3], dtype=np.int32)
    logit_length = np.array([15, 15, 15], dtype=np.int32)
    logits_time_major = False
    unique = np.array([], dtype=np.int32)
    blank_index = 5
    name = "ctc_loss_4"
    list_of_inputs.append({
        'labels': labels,
        'logits': logits,
        'label_length': label_length,
        'logit_length': logit_length,
        'logits_time_major': logits_time_major,
        'unique': unique,
        'blank_index': blank_index,
        'name': name
    })

    # Input 5
    labels = np.array([[1, 2, 3, 4, 5, 0], [5, 4, 3, 2, 1, 0], [2, 3, 4, 0, 0, 0]], dtype=np.int64)
    logits = np.random.uniform(size=(12, 3, 8)).astype(np.float32)
    label_length = np.array([5, 5, 3], dtype=np.int64)
    logit_length = np.array([12, 12, 12], dtype=np.int64)
    logits_time_major = True
    unique = np.array([], dtype=np.int64)
    blank_index = 0
    name = "ctc_loss_5"
    list_of_inputs.append({
        'labels': labels,
        'logits': logits,
        'label_length': label_length,
        'logit_length': logit_length,
        'logits_time_major': logits_time_major,
        'unique': unique,
        'blank_index': blank_index,
        'name': name
    })

    # Input 6
    labels = np.random.randint(1, 5, size=(16, 8)).astype(np.int32)
    logits = np.random.uniform(size=(16, 20, 10)).astype(np.float32)
    label_length = np.random.randint(4, 9, size=16).astype(np.int32)
    logit_length = np.random.randint(15, 21, size=16).astype(np.int32)
    logits_time_major = False
    unique = np.array([], dtype=np.int32)
    blank_index = 0
    name = "ctc_loss_6"
    list_of_inputs.append({
        'labels': labels,
        'logits': logits,
        'label_length': label_length,
        'logit_length': logit_length,
        'logits_time_major': logits_time_major,
        'unique': unique,
        'blank_index': blank_index,
        'name': name
    })

    # Input 7
    labels = np.random.randint(1, 15, size=(5, 10)).astype(np.int32)
    logits = np.random.uniform(size=(30, 5, 20)).astype(np.float32)
    label_length = np.array([8, 9, 7, 10, 6], dtype=np.int32)
    logit_length = np.array([28, 29, 30, 27, 26], dtype=np.int32)
    logits_time_major = True
    unique = np.array([], dtype=np.int32)
    blank_index = 19
    name = "ctc_loss_7"
    list_of_inputs.append({
        'labels': labels,
        'logits': logits,
        'label_length': label_length,
        'logit_length': logit_length,
        'logits_time_major': logits_time_major,
        'unique': unique,
        'blank_index': blank_index,
        'name': name
    })

    # Input 8
    labels = np.array([[0], [0]], dtype=np.int64)
    logits = np.random.uniform(size=(2, 2, 2)).astype(np.float32)
    label_length = np.array([1, 1], dtype=np.int64)
    logit_length = np.array([2, 2], dtype=np.int64)
    logits_time_major = False
    unique = np.array([], dtype=np.int64)
    blank_index = 1
    name = "ctc_loss_8"
    list_of_inputs.append({
        'labels': labels,
        'logits': logits,
        'label_length': label_length,
        'logit_length': logit_length,
        'logits_time_major': logits_time_major,
        'unique': unique,
        'blank_index': blank_index,
        'name': name
    })

    # Input 9
    labels = np.random.randint(1, 5, size=(10, 5)).astype(np.int32)
    logits = np.random.uniform(size=(15, 10, 6)).astype(np.float32)
    label_length = np.array([4, 3, 5, 4, 3, 5, 4, 3, 5, 4], dtype=np.int32)
    logit_length = np.array([15] * 10, dtype=np.int32)
    logits_time_major = True
    unique = np.array([], dtype=np.int32)
    blank_index = -1
    name = "ctc_loss_9"
    list_of_inputs.append({
        'labels': labels,
        'logits': logits,
        'label_length': label_length,
        'logit_length': logit_length,
        'logits_time_major': logits_time_major,
        'unique': unique,
        'blank_index': blank_index,
        'name': name
    })

    # Input 10
    labels = np.random.randint(1, 3, size=(4, 3)).astype(np.int64)
    logits = np.random.uniform(size=(4, 6, 4)).astype(np.float32)
    label_length = np.array([3, 2, 3, 2], dtype=np.int64)
    logit_length = np.array([6, 5, 6, 5], dtype=np.int64)
    logits_time_major = False
    unique = np.array([], dtype=np.int64)
    blank_index = 0
    name = "ctc_loss_10"
    list_of_inputs.append({
        'labels': labels,
        'logits': logits,
        'label_length': label_length,
        'logit_length': logit_length,
        'logits_time_major': logits_time_major,
        'unique': unique,
        'blank_index': blank_index,
        'name': name
    })

    return list_of_inputs

generated_inputs["tf.nn.ctc_loss"] = tf_nn_ctc_loss_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_weighted_cross_entropy_with_logits_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, pos_weight scalar
    labels = np.array([1.0, 0.5, 0.0], dtype=np.float32)
    logits = np.array([1.5, -0.1, -10.0], dtype=np.float32)
    pos_weight = np.array(1.5, dtype=np.float32)
    name = "loss_1"
    list_of_inputs.append({
        "labels": labels,
        "logits": logits,
        "pos_weight": pos_weight,
        "name": name
    })

    # Input 2: 1D array, float32, pos_weight scalar < 1.0
    labels = np.array([0.1, 0.9, 0.5], dtype=np.float32)
    logits = np.array([0.0, 2.0, -1.0], dtype=np.float32)
    pos_weight = np.array(0.5, dtype=np.float32)
    name = "loss_2"
    list_of_inputs.append({
        "labels": labels,
        "logits": logits,
        "pos_weight": pos_weight,
        "name": name
    })

    # Input 3: 2D array, float32, pos_weight scalar
    labels = np.array([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    logits = np.array([[-1.0, 1.0], [0.0, -2.0]], dtype=np.float32)
    pos_weight = np.array(2.0, dtype=np.float32)
    name = "loss_3"
    list_of_inputs.append({
        "labels": labels,
        "logits": logits,
        "pos_weight": pos_weight,
        "name": name
    })

    # Input 4: 2D array, float64, pos_weight scalar
    labels = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    logits = np.array([[10.0, -10.0], [-5.0, 5.0]], dtype=np.float64)
    pos_weight = np.array(1.0, dtype=np.float64)
    name = "loss_4"
    list_of_inputs.append({
        "labels": labels,
        "logits": logits,
        "pos_weight": pos_weight,
        "name": name
    })

    # Input 5: 3D array, float32, pos_weight 1D array (broadcastable)
    labels = np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32)
    logits = np.array([[[1.0, -1.0], [2.0, -2.0]]], dtype=np.float32)
    pos_weight = np.array([1.5, 0.5], dtype=np.float32)
    name = "loss_5"
    list_of_inputs.append({
        "labels": labels,
        "logits": logits,
        "pos_weight": pos_weight,
        "name": name
    })

    # Input 6: 1D array, float64, pos_weight broadcastable 1D array
    labels = np.array([0.0, 1.0, 0.5, 0.8], dtype=np.float64)
    logits = np.array([-100.0, 100.0, 0.0, -1.0], dtype=np.float64)
    pos_weight = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    name = "loss_6"
    list_of_inputs.append({
        "labels": labels,
        "logits": logits,
        "pos_weight": pos_weight,
        "name": name
    })

    # Input 7: 2D array, float32, pos_weight broadcastable 2D array
    labels = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    logits = np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=np.float32)
    pos_weight = np.array([[1.0], [2.0]], dtype=np.float32)
    name = "loss_7"
    list_of_inputs.append({
        "labels": labels,
        "logits": logits,
        "pos_weight": pos_weight,
        "name": name
    })

    # Input 8: 1D array with single element, float32
    labels = np.array([0.0], dtype=np.float32)
    logits = np.array([-0.5], dtype=np.float32)
    pos_weight = np.array([3.5], dtype=np.float32)
    name = "loss_8"
    list_of_inputs.append({
        "labels": labels,
        "logits": logits,
        "pos_weight": pos_weight,
        "name": name
    })

    # Input 9: 3D array, float64, pos_weight scalar
    labels = np.array([[[0.0], [1.0]], [[1.0], [0.0]]], dtype=np.float64)
    logits = np.array([[[0.1], [-0.1]], [[0.2], [-0.2]]], dtype=np.float64)
    pos_weight = np.array(0.1, dtype=np.float64)
    name = "loss_9"
    list_of_inputs.append({
        "labels": labels,
        "logits": logits,
        "pos_weight": pos_weight,
        "name": name
    })

    # Input 10: 2D array, float32, pos_weight matching shape
    labels = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    logits = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    pos_weight = np.array([[0.5, 1.0, 1.5]], dtype=np.float32)
    name = "loss_10"
    list_of_inputs.append({
        "labels": labels,
        "logits": logits,
        "pos_weight": pos_weight,
        "name": name
    })

    return list_of_inputs

generated_inputs["tf.nn.weighted_cross_entropy_with_logits"] = tf_weighted_cross_entropy_with_logits_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_one_hot_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "indices": np.array([0, 1, 2], dtype=np.int32),
        "depth": 3,
        "on_value": np.array(1.0, dtype=np.float32),
        "off_value": np.array(0.0, dtype=np.float32),
        "axis": -1,
        "dtype": np.float32,
        "name": "one_hot_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "indices": np.array([[0, 2], [1, -1]], dtype=np.int32),
        "depth": 3,
        "on_value": np.array(1.0, dtype=np.float32),
        "off_value": np.array(0.0, dtype=np.float32),
        "axis": -1,
        "dtype": np.float32,
        "name": "one_hot_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "indices": np.array([1, 0], dtype=np.int32),
        "depth": 2,
        "on_value": np.array(5.0, dtype=np.float64),
        "off_value": np.array(-1.0, dtype=np.float64),
        "axis": 0,
        "dtype": np.float64,
        "name": "one_hot_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "indices": np.array([[[0, 1], [2, 3]]], dtype=np.int64),
        "depth": 4,
        "on_value": np.array(1, dtype=np.int32),
        "off_value": np.array(0, dtype=np.int32),
        "axis": -1,
        "dtype": np.int32,
        "name": "one_hot_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "indices": np.array(2, dtype=np.int32),
        "depth": 5,
        "on_value": np.array(1.0, dtype=np.float32),
        "off_value": np.array(0.0, dtype=np.float32),
        "axis": -1,
        "dtype": np.float32,
        "name": "one_hot_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "indices": np.array([0, -1, 2], dtype=np.int32),
        "depth": 3,
        "on_value": np.array(True, dtype=np.bool_),
        "off_value": np.array(False, dtype=np.bool_),
        "axis": -1,
        "dtype": np.bool_,
        "name": "one_hot_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "indices": np.array([[1], [0]], dtype=np.int32),
        "depth": 2,
        "on_value": np.array(2.5, dtype=np.float32),
        "off_value": np.array(0.1, dtype=np.float32),
        "axis": 1,
        "dtype": np.float32,
        "name": "one_hot_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "indices": np.array([2, 1, 0], dtype=np.int32),
        "depth": 3,
        "on_value": np.array(1, dtype=np.int64),
        "off_value": np.array(0, dtype=np.int64),
        "axis": 0,
        "dtype": np.int64,
        "name": "one_hot_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "indices": np.array([[[1]]], dtype=np.int32),
        "depth": 2,
        "on_value": np.array(10.0, dtype=np.float32),
        "off_value": np.array(-10.0, dtype=np.float32),
        "axis": 2,
        "dtype": np.float32,
        "name": "one_hot_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "indices": np.array([0], dtype=np.int32),
        "depth": 1,
        "on_value": np.array(1, dtype=np.int32),
        "off_value": np.array(0, dtype=np.int32),
        "axis": -1,
        "dtype": np.int32,
        "name": "one_hot_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.one_hot"] = tf_one_hot_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_pad_inputs():
    list_of_inputs = []

    # Input 1: 1D CONSTANT pad with float32
    list_of_inputs.append({
        'tensor': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'paddings': np.array([[1, 2]], dtype=np.int32),
        'mode': "CONSTANT",
        'constant_values': np.array(0.0, dtype=np.float32),
        'name': "pad_1"
    })

    # Input 2: 2D CONSTANT pad with negative fill, int32
    list_of_inputs.append({
        'tensor': np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        'paddings': np.array([[1, 1], [2, 2]], dtype=np.int32),
        'mode': "CONSTANT",
        'constant_values': np.array(-1, dtype=np.int32),
        'name': "pad_2"
    })

    # Input 3: 2D REFLECT pad
    list_of_inputs.append({
        'tensor': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        'paddings': np.array([[1, 1], [2, 2]], dtype=np.int32),
        'mode': "REFLECT",
        'constant_values': np.array(0.0, dtype=np.float32),
        'name': "pad_3"
    })

    # Input 4: 2D SYMMETRIC pad, float64
    list_of_inputs.append({
        'tensor': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
        'paddings': np.array([[2, 2], [3, 3]], dtype=np.int32),
        'mode': "SYMMETRIC",
        'constant_values': np.array(0.0, dtype=np.float64),
        'name': "pad_4"
    })

    # Input 5: 3D CONSTANT pad
    list_of_inputs.append({
        'tensor': np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        'paddings': np.array([[1, 0], [0, 2], [1, 1]], dtype=np.int32),
        'mode': "CONSTANT",
        'constant_values': np.array(0.5, dtype=np.float32),
        'name': "pad_5"
    })

    # Input 6: 4D CONSTANT pad, int64
    list_of_inputs.append({
        'tensor': np.ones((2, 2, 2, 2), dtype=np.int64),
        'paddings': np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=np.int32),
        'mode': "CONSTANT",
        'constant_values': np.array(9, dtype=np.int64),
        'name': "pad_6"
    })

    # Input 7: 1D REFLECT pad
    list_of_inputs.append({
        'tensor': np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        'paddings': np.array([[3, 3]], dtype=np.int32),
        'mode': "REFLECT",
        'constant_values': np.array(0.0, dtype=np.float32),
        'name': "pad_7"
    })

    # Input 8: 3D SYMMETRIC pad
    list_of_inputs.append({
        'tensor': np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        'paddings': np.array([[1, 1], [2, 2], [3, 3]], dtype=np.int32),
        'mode': "SYMMETRIC",
        'constant_values': np.array(0.0, dtype=np.float32),
        'name': "pad_8"
    })

    # Input 9: 2D Zero pad
    list_of_inputs.append({
        'tensor': np.array([[1.5]], dtype=np.float32),
        'paddings': np.array([[0, 0], [0, 0]], dtype=np.int32),
        'mode': "CONSTANT",
        'constant_values': np.array(0.0, dtype=np.float32),
        'name': "pad_9"
    })

    # Input 10: Boolean tensor pad
    list_of_inputs.append({
        'tensor': np.array([True, False], dtype=bool),
        'paddings': np.array([[1, 1]], dtype=np.int32),
        'mode': "CONSTANT",
        'constant_values': np.array(True, dtype=bool),
        'name': "pad_10"
    })

    return list_of_inputs

generated_inputs["tf.pad"] = tf_pad_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_quantization_fake_quant_with_min_max_vars_gradient_inputs():
    list_of_inputs = []

    # Case 1: 1D input, 8 bits, standard range
    input_dict = {
        "gradients": np.array([0.1, -0.2, 0.3], dtype=np.float32),
        "inputs": np.array([0.5, -0.5, 1.2], dtype=np.float32),
        "min": np.array(-1.0, dtype=np.float32),
        "max": np.array(1.0, dtype=np.float32),
        "num_bits": 8,
        "narrow_range": False,
        "name": "case1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D input, 4 bits, narrow range
    input_dict = {
        "gradients": np.array([[0.5, -0.1], [0.2, -0.3]], dtype=np.float32),
        "inputs": np.array([[1.5, -1.5], [0.5, -0.5]], dtype=np.float32),
        "min": np.array(-2.0, dtype=np.float32),
        "max": np.array(2.0, dtype=np.float32),
        "num_bits": 4,
        "narrow_range": True,
        "name": "case2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 3D input, 2 bits, standard range
    input_dict = {
        "gradients": np.ones((2, 2, 2), dtype=np.float32) * 0.1,
        "inputs": np.zeros((2, 2, 2), dtype=np.float32),
        "min": np.array(-0.5, dtype=np.float32),
        "max": np.array(0.5, dtype=np.float32),
        "num_bits": 2,
        "narrow_range": False,
        "name": "case3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 4D input, 7 bits, narrow range
    input_dict = {
        "gradients": np.random.randn(1, 2, 2, 3).astype(np.float32),
        "inputs": np.random.randn(1, 2, 2, 3).astype(np.float32),
        "min": np.array(-3.0, dtype=np.float32),
        "max": np.array(3.0, dtype=np.float32),
        "num_bits": 7,
        "narrow_range": True,
        "name": "case4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 1D size 1 input, 8 bits
    input_dict = {
        "gradients": np.array([1.5], dtype=np.float32),
        "inputs": np.array([0.0], dtype=np.float32),
        "min": np.array(-1.0, dtype=np.float32),
        "max": np.array(1.0, dtype=np.float32),
        "num_bits": 8,
        "narrow_range": False,
        "name": "case5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Large values, 5 bits, standard range
    input_dict = {
        "gradients": np.array([[10., 20.], [-10., -20.]], dtype=np.float32),
        "inputs": np.array([[5., 15.], [-5., -15.]], dtype=np.float32),
        "min": np.array(-10.0, dtype=np.float32),
        "max": np.array(10.0, dtype=np.float32),
        "num_bits": 5,
        "narrow_range": False,
        "name": "case6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Positive min and max, 8 bits
    input_dict = {
        "gradients": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "inputs": np.array([[1.5, 2.5], [2.0, 3.5]], dtype=np.float32),
        "min": np.array(1.0, dtype=np.float32),
        "max": np.array(3.0, dtype=np.float32),
        "num_bits": 8,
        "narrow_range": False,
        "name": "case7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: Negative min and max, 6 bits, narrow range
    input_dict = {
        "gradients": np.array([[-0.1, -0.2], [-0.3, -0.4]], dtype=np.float32),
        "inputs": np.array([[-1.5, -2.5], [-2.0, -3.5]], dtype=np.float32),
        "min": np.array(-5.0, dtype=np.float32),
        "max": np.array(-1.0, dtype=np.float32),
        "num_bits": 6,
        "narrow_range": True,
        "name": "case8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: Very small range, 3 bits
    input_dict = {
        "gradients": np.array([0.01, -0.02, 0.03, -0.04, 0.05], dtype=np.float32),
        "inputs": np.array([0.005, -0.005, 0.012, -0.015, 0.02], dtype=np.float32),
        "min": np.array(-0.01, dtype=np.float32),
        "max": np.array(0.01, dtype=np.float32),
        "num_bits": 3,
        "narrow_range": False,
        "name": "case9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: 1D array of size 10, zero gradients, 8 bits
    input_dict = {
        "gradients": np.zeros((10,), dtype=np.float32),
        "inputs": np.ones((10,), dtype=np.float32),
        "min": np.array(0.0, dtype=np.float32),
        "max": np.array(1.0, dtype=np.float32),
        "num_bits": 8,
        "narrow_range": True,
        "name": "case10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.quantization.fake_quant_with_min_max_vars_gradient"] = tf_quantization_fake_quant_with_min_max_vars_gradient_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_random_stateless_binomial_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'shape': np.array([2], dtype=np.int32),
        'seed': np.array([123, 456], dtype=np.int32),
        'counts': np.array([10., 20.], dtype=np.float32),
        'probs': np.array([0.8, 0.5], dtype=np.float32),
        'output_dtype': np.int32,
        'name': 'binom_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'shape': np.array([5], dtype=np.int32),
        'seed': np.array([789, 1011], dtype=np.int32),
        'counts': np.array([10., 10., 10., 10., 10.], dtype=np.float32),
        'probs': np.array([0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32),
        'output_dtype': np.int64,
        'name': 'binom_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'shape': np.array([3, 2], dtype=np.int32),
        'seed': np.array([1, 2], dtype=np.int64),
        'counts': np.array([[5., 15.], [5., 15.], [5., 15.]], dtype=np.float32),
        'probs': np.array([[0.2, 0.7], [0.2, 0.7], [0.2, 0.7]], dtype=np.float32),
        'output_dtype': np.int32,
        'name': 'binom_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'shape': np.array([4, 1], dtype=np.int32),
        'seed': np.array([42, 42], dtype=np.int32),
        'counts': np.array([[100.], [100.], [100.], [100.]], dtype=np.float32),
        'probs': np.array([[0.5], [0.5], [0.5], [0.5]], dtype=np.float32),
        'output_dtype': np.int32,
        'name': 'binom_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'shape': np.array([3], dtype=np.int64),
        'seed': np.array([-1, -2], dtype=np.int32),
        'counts': np.array([50., 50., 50.], dtype=np.float64),
        'probs': np.array([0.1, 0.2, 0.3], dtype=np.float64),
        'output_dtype': np.int64,
        'name': 'binom_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'shape': np.array([2, 2], dtype=np.int32),
        'seed': np.array([0, 0], dtype=np.int32),
        'counts': np.array([[10., 20.], [30., 40.]], dtype=np.float32),
        'probs': np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
        'output_dtype': np.int32,
        'name': 'binom_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'shape': np.array([1], dtype=np.int32),
        'seed': np.array([1000, 2000], dtype=np.int64),
        'counts': np.array([1.], dtype=np.float32),
        'probs': np.array([0.99], dtype=np.float32),
        'output_dtype': np.int32,
        'name': 'binom_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'shape': np.array([4], dtype=np.int32),
        'seed': np.array([11, 22], dtype=np.int32),
        'counts': np.array([5., 10., 15., 20.], dtype=np.float32),
        'probs': np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        'output_dtype': np.int64,
        'name': 'binom_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'shape': np.array([2, 3], dtype=np.int32),
        'seed': np.array([99, 99], dtype=np.int32),
        'counts': np.array([[10., 20., 30.], [40., 50., 60.]], dtype=np.float32),
        'probs': np.array([[0.4, 0.4, 0.4], [0.4, 0.4, 0.4]], dtype=np.float32),
        'output_dtype': np.int32,
        'name': 'binom_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'shape': np.array([1, 1], dtype=np.int32),
        'seed': np.array([8888, 9999], dtype=np.int64),
        'counts': np.array([[5.]], dtype=np.float32),
        'probs': np.array([[0.0]], dtype=np.float32),
        'output_dtype': np.int32,
        'name': 'binom_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random.stateless_binomial"] = tf_random_stateless_binomial_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_random_stateless_gamma_inputs():
    list_of_inputs = []

    # Input 1
    input_dict_1 = {
        "shape": np.array([5], dtype=np.int32),
        "seed": np.array([42, 43], dtype=np.int32),
        "alpha": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "beta": np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "dtype": np.float32,
        "name": "gamma_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2
    input_dict_2 = {
        "shape": np.array([10, 2], dtype=np.int32),
        "seed": np.array([12, 34], dtype=np.int32),
        "alpha": np.array([0.5, 1.5], dtype=np.float32),
        "beta": np.array([1.0, 2.0], dtype=np.float32),
        "dtype": np.float32,
        "name": "gamma_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3
    input_dict_3 = {
        "shape": np.array([3, 4], dtype=np.int32),
        "seed": np.array([7, 8], dtype=np.int64),
        "alpha": np.array([[1.0], [2.0], [3.0]], dtype=np.float64),
        "beta": np.array([[2.0, 3.0, 4.0, 5.0]], dtype=np.float64),
        "dtype": np.float64,
        "name": "gamma_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4
    input_dict_4 = {
        "shape": np.array([5, 3, 4], dtype=np.int32),
        "seed": np.array([1, 2], dtype=np.int32),
        "alpha": np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        "beta": np.array([[2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
        "dtype": np.float32,
        "name": "gamma_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5
    input_dict_5 = {
        "shape": np.array([1], dtype=np.int32),
        "seed": np.array([100, 200], dtype=np.int32),
        "alpha": np.array([2.5], dtype=np.float32),
        "beta": np.array([1.0], dtype=np.float32),
        "dtype": np.float32,
        "name": "gamma_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6
    input_dict_6 = {
        "shape": np.array([2, 3], dtype=np.int32),
        "seed": np.array([999, 888], dtype=np.int32),
        "alpha": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "beta": np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32),
        "dtype": np.float32,
        "name": "gamma_large_alpha"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7
    input_dict_7 = {
        "shape": np.array([4, 2, 3], dtype=np.int32),
        "seed": np.array([11, 22], dtype=np.int32),
        "alpha": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float16),
        "beta": np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float16),
        "dtype": np.float16,
        "name": "gamma_float16"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8
    input_dict_8 = {
        "shape": np.array([2, 2], dtype=np.int32),
        "seed": np.array([3, 4], dtype=np.int64),
        "alpha": np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32),
        "beta": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        "dtype": np.float32,
        "name": "gamma_4d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9
    input_dict_9 = {
        "shape": np.array([1, 1], dtype=np.int32),
        "seed": np.array([123, 456], dtype=np.int32),
        "alpha": np.array([[1.0]], dtype=np.float32),
        "beta": np.array([[1.0]], dtype=np.float32),
        "dtype": np.float32,
        "name": "gamma_scalar"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10
    input_dict_10 = {
        "shape": np.array([2, 1, 3], dtype=np.int32),
        "seed": np.array([123456789, 987654321], dtype=np.int64),
        "alpha": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        "beta": np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
        "dtype": np.float32,
        "name": "gamma_large_seed"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.random.stateless_gamma"] = tf_random_stateless_gamma_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_random_stateless_parameterized_truncated_normal_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D shape with 0-D parameters
    input_dict = {
        'shape': np.array([5], dtype=np.int32),
        'seed': np.array([1, 2], dtype=np.int32),
        'means': np.array(0.0, dtype=np.float32),
        'stddevs': np.array(1.0, dtype=np.float32),
        'minvals': np.array(-2.0, dtype=np.float32),
        'maxvals': np.array(2.0, dtype=np.float32),
        'name': 'simple_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D shape with suffix-matching 1D broadcasted parameter
    input_dict = {
        'shape': np.array([3, 4], dtype=np.int32),
        'seed': np.array([42, 43], dtype=np.int32),
        'means': np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        'stddevs': np.array(0.5, dtype=np.float32),
        'minvals': np.array(-1.0, dtype=np.float32),
        'maxvals': np.array(5.0, dtype=np.float32),
        'name': 'broadcast_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float64 parameters with matching 2D shape suffix
    input_dict = {
        'shape': np.array([10, 1], dtype=np.int64),
        'seed': np.array([100, 200], dtype=np.int64),
        'means': np.array([-5.0], dtype=np.float64),
        'stddevs': np.array([2.5], dtype=np.float64),
        'minvals': np.array([-10.0], dtype=np.float64),
        'maxvals': np.array([0.0], dtype=np.float64),
        'name': 'float64_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D shape, multi-dimensional broadcasting resulting in exact suffix [3, 4]
    input_dict = {
        'shape': np.array([2, 3, 4], dtype=np.int32),
        'seed': np.array([5, 5], dtype=np.int32),
        'means': np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        'stddevs': np.array([1.0, 1.5, 2.0, 2.5], dtype=np.float32),
        'minvals': np.array(-5.0, dtype=np.float32),
        'maxvals': np.array(5.0, dtype=np.float32),
        'name': 'multi_broadcast_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Extremely tight boundaries with 0-D parameters
    input_dict = {
        'shape': np.array([100], dtype=np.int32),
        'seed': np.array([7, 11], dtype=np.int32),
        'means': np.array(0.0, dtype=np.float32),
        'stddevs': np.array(1.0, dtype=np.float32),
        'minvals': np.array(-0.1, dtype=np.float32),
        'maxvals': np.array(0.1, dtype=np.float32),
        'name': 'tight_bounds'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large standard deviation and matched exact-shape parameters
    input_dict = {
        'shape': np.array([2, 2], dtype=np.int32),
        'seed': np.array([12, 34], dtype=np.int32),
        'means': np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        'stddevs': np.array([[100.0, 100.0], [100.0, 100.0]], dtype=np.float32),
        'minvals': np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        'maxvals': np.array([[50.0, 50.0], [50.0, 50.0]], dtype=np.float32),
        'name': 'large_stddev'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: One-sided truncation with 0-D parameters
    input_dict = {
        'shape': np.array([5, 5], dtype=np.int32),
        'seed': np.array([99, 99], dtype=np.int32),
        'means': np.array(0.0, dtype=np.float32),
        'stddevs': np.array(1.0, dtype=np.float32),
        'minvals': np.array(0.0, dtype=np.float32),
        'maxvals': np.array(1000.0, dtype=np.float32),
        'name': 'one_sided_truncation'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Deeply nested shape with exact suffix matching 3D tensors
    input_dict = {
        'shape': np.array([2, 2, 2, 2], dtype=np.int32),
        'seed': np.array([1, 1], dtype=np.int32),
        'means': np.zeros((2, 2, 2), dtype=np.float32),
        'stddevs': np.ones((2, 2, 2), dtype=np.float32),
        'minvals': np.ones((2, 2, 2), dtype=np.float32) * -3.0,
        'maxvals': np.ones((2, 2, 2), dtype=np.float32) * 3.0,
        'name': 'deep_4d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Suffix-matching 1D parameter broadcasted to shape [8]
    input_dict = {
        'shape': np.array([8], dtype=np.int32),
        'seed': np.array([456, 789], dtype=np.int32),
        'means': np.array(-10.0, dtype=np.float32),
        'stddevs': np.array(5.0, dtype=np.float32),
        'minvals': np.array(-20.0, dtype=np.float32),
        'maxvals': np.array([-5.0] * 8, dtype=np.float32),
        'name': 'negative_means'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D shape with 0-D parameters
    input_dict = {
        'shape': np.array([3, 3, 3], dtype=np.int32),
        'seed': np.array([0, 0], dtype=np.int32),
        'means': np.array(5.0, dtype=np.float32),
        'stddevs': np.array(0.1, dtype=np.float32),
        'minvals': np.array(4.8, dtype=np.float32),
        'maxvals': np.array(5.2, dtype=np.float32),
        'name': 'scalar_broadcast'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random.stateless_parameterized_truncated_normal"] = tf_random_stateless_parameterized_truncated_normal_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_AdjustSaturation_inputs():
    list_of_inputs = []
    
    # Input 1
    images = np.random.rand(2, 2, 3).astype(np.float32)
    scale = np.array(0.5, dtype=np.float32)
    name = "adjust_sat_1"
    list_of_inputs.append({"name": name, "images": images, "scale": scale})

    # Input 2
    images = np.random.rand(1, 3, 3, 3).astype(np.float32)
    scale = np.array(-0.2, dtype=np.float32)
    name = "adjust_sat_2"
    list_of_inputs.append({"name": name, "images": images, "scale": scale})

    # Input 3
    images = np.random.rand(2, 2, 3).astype(np.float32)
    scale = np.array(1.0, dtype=np.float32)
    name = "adjust_sat_3"
    list_of_inputs.append({"name": name, "images": images, "scale": scale})

    # Input 4
    images = np.random.rand(4, 4, 3).astype(np.float32)
    scale = np.array(-1.5, dtype=np.float32)
    name = "adjust_sat_4"
    list_of_inputs.append({"name": name, "images": images, "scale": scale})

    # Input 5
    images = np.random.rand(2, 3, 4, 3).astype(np.float32)
    scale = np.array(0.0, dtype=np.float32)
    name = "adjust_sat_5"
    list_of_inputs.append({"name": name, "images": images, "scale": scale})

    # Input 6
    images = np.random.rand(5, 5, 5, 3).astype(np.float32)
    scale = np.array(2.5, dtype=np.float32)
    name = "adjust_sat_6"
    list_of_inputs.append({"name": name, "images": images, "scale": scale})

    # Input 7
    images = np.random.rand(1, 1, 3).astype(np.float32)
    scale = np.array(-0.5, dtype=np.float32)
    name = "adjust_sat_7"
    list_of_inputs.append({"name": name, "images": images, "scale": scale})

    # Input 8
    images = np.random.rand(3, 3, 3).astype(np.float32)
    scale = np.array(0.75, dtype=np.float32)
    name = "adjust_sat_8"
    list_of_inputs.append({"name": name, "images": images, "scale": scale})

    # Input 9
    images = np.random.rand(1, 2, 3, 4, 3).astype(np.float32)
    scale = np.array(1.2, dtype=np.float32)
    name = "adjust_sat_9"
    list_of_inputs.append({"name": name, "images": images, "scale": scale})

    # Input 10
    images = np.random.rand(10, 10, 3).astype(np.float32)
    scale = np.array(-2.0, dtype=np.float32)
    name = "adjust_sat_10"
    list_of_inputs.append({"name": name, "images": images, "scale": scale})

    return list_of_inputs

generated_inputs["tf.raw_ops.AdjustSaturation"] = tf_raw_ops_AdjustSaturation_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ArgMax_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'input': np.array([1.0, 10.0, 26.9, 2.8, 166.32, 62.3], dtype=np.float32),
        'dimension': np.array(0, dtype=np.int32),
        'output_type': tf.int64,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'input': np.array([[1, 5, 3], [4, 2, 6]], dtype=np.int32),
        'dimension': np.array(1, dtype=np.int32),
        'output_type': tf.int32,
        'name': 'argmax_2d_axis1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'input': np.array([[[1.5, 2.3], [4.1, 0.2]], [[-1.2, 5.5], [0.0, -3.1]]], dtype=np.float64),
        'dimension': np.array(2, dtype=np.int64),
        'output_type': tf.int64,
        'name': 'argmax_3d_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'input': np.random.randint(-100, 100, size=(2, 3, 4, 5), dtype=np.int64),
        'dimension': np.array(-1, dtype=np.int32),
        'output_type': tf.int32,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'input': np.array([-10.5, -50.2, -2.1, -100.0], dtype=np.float32),
        'dimension': np.array(-1, dtype=np.int32),
        'output_type': tf.int32,
        'name': 'argmax_1d_neg'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'input': np.array([[-1.5, -2.5], [3.5, 1.2]], dtype=np.float64),
        'dimension': np.array(0, dtype=np.int32),
        'output_type': tf.int64,
        'name': 'argmax_2d_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'input': np.random.uniform(-10.0, 10.0, size=(2, 2, 2)).astype(np.float32),
        'dimension': np.array(-2, dtype=np.int32),
        'output_type': tf.int32,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'input': np.random.randint(-100, 100, size=(2, 2, 3, 2, 2), dtype=np.int64),
        'dimension': np.array(3, dtype=np.int64),
        'output_type': tf.int64,
        'name': 'argmax_5d_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'input': np.array([[10, 20], [40, 30]], dtype=np.int32),
        'dimension': np.array(1, dtype=np.int64),
        'output_type': tf.int32,
        'name': 'argmax_int32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'input': np.array([0.1, 0.9, 0.4, 0.5], dtype=np.float32),
        'dimension': np.array(0, dtype=np.int32),
        'output_type': tf.int64,
        'name': 'argmax_float32_decimals'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ArgMax"] = tf_raw_ops_ArgMax_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ArgMin_inputs():
    list_of_inputs = []
    
    # Input 1
    list_of_inputs.append({
        "input": np.array([1.0, 10.0, 26.9, 2.8, 166.32, 62.3], dtype=np.float32),
        "dimension": np.array(0, dtype=np.int32),
        "output_type": tf.int64,
        "name": "argmin_1"
    })
    
    # Input 2
    list_of_inputs.append({
        "input": np.array([[1, 2, 3], [4, 5, -1]], dtype=np.int32),
        "dimension": np.array(1, dtype=np.int32),
        "output_type": tf.int32,
        "name": "argmin_2"
    })

    # Input 3
    list_of_inputs.append({
        "input": np.array([[[10], [20]], [[30], [5]], [[40], [50]]], dtype=np.uint8),
        "dimension": np.array(0, dtype=np.int64),
        "output_type": tf.int64,
        "name": "argmin_3"
    })

    # Input 4
    list_of_inputs.append({
        "input": np.array([-1.5, -2.5, -0.5, -10.2], dtype=np.float64),
        "dimension": np.array(0, dtype=np.int32),
        "output_type": tf.int32,
        "name": "argmin_4"
    })

    # Input 5
    list_of_inputs.append({
        "input": np.array([[True, False], [False, False]], dtype=np.bool_),
        "dimension": np.array(0, dtype=np.int32),
        "output_type": tf.int64,
        "name": "argmin_5"
    })

    # Input 6
    list_of_inputs.append({
        "input": np.array([[10, 20, 30], [5, 2, 1]], dtype=np.int16),
        "dimension": np.array(1, dtype=np.int64),
        "output_type": tf.int32,
        "name": "argmin_6"
    })

    # Input 7
    list_of_inputs.append({
        "input": np.random.randint(-100, 100, size=(2, 3, 4)).astype(np.int64),
        "dimension": np.array(2, dtype=np.int32),
        "output_type": tf.int64,
        "name": "argmin_7"
    })

    # Input 8
    list_of_inputs.append({
        "input": np.random.uniform(-10.0, 10.0, size=(5,)).astype(np.float32),
        "dimension": np.array(-1, dtype=np.int32),
        "output_type": tf.int32,
        "name": "argmin_8"
    })

    # Input 9
    list_of_inputs.append({
        "input": np.array([[3, 3, 3], [3, 3, 3]], dtype=np.int32),
        "dimension": np.array(1, dtype=np.int32),
        "output_type": tf.int64,
        "name": "argmin_9"
    })

    # Input 10
    list_of_inputs.append({
        "input": np.ones((2, 2, 2, 2), dtype=np.float32),
        "dimension": np.array(3, dtype=np.int32),
        "output_type": tf.int32,
        "name": "argmin_10"
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.ArgMin"] = tf_raw_ops_ArgMin_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_bincount_inputs():
    list_of_inputs = []

    # Input 1: Basic case, 1D arr, size 5, empty float32 weights
    list_of_inputs.append({
        "name": "bincount_1",
        "arr": np.array([1, 2, 2, 3], dtype=np.int32),
        "size": np.array(5, dtype=np.int32),
        "weights": np.array([], dtype=np.float32)
    })

    # Input 2: 1D arr, size 4, with matching float32 weights
    list_of_inputs.append({
        "name": "bincount_2",
        "arr": np.array([0, 1, 2, 1], dtype=np.int32),
        "size": np.array(4, dtype=np.int32),
        "weights": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    })

    # Input 3: Values outside range [0, size) to be ignored
    list_of_inputs.append({
        "name": "bincount_3",
        "arr": np.array([0, 5, 2, 10], dtype=np.int32),
        "size": np.array(4, dtype=np.int32),
        "weights": np.array([], dtype=np.int32)
    })

    # Input 4: Using int64 weights
    list_of_inputs.append({
        "name": "bincount_4",
        "arr": np.array([0, 1, 2], dtype=np.int32),
        "size": np.array(3, dtype=np.int32),
        "weights": np.array([10, 20, 30], dtype=np.int64)
    })

    # Input 5: Using float64 weights
    list_of_inputs.append({
        "name": "bincount_5",
        "arr": np.array([1, 1, 1], dtype=np.int32),
        "size": np.array(2, dtype=np.int32),
        "weights": np.array([1.5, 2.5, 3.5], dtype=np.float64)
    })

    # Input 6: Empty arr and float32 empty weights
    list_of_inputs.append({
        "name": "bincount_6",
        "arr": np.array([], dtype=np.int32),
        "size": np.array(3, dtype=np.int32),
        "weights": np.array([], dtype=np.float32)
    })

    # Input 7: Values in arr outside [0, size) (must be non-negative)
    list_of_inputs.append({
        "name": "bincount_7",
        "arr": np.array([5, 10, 15], dtype=np.int32),
        "size": np.array(2, dtype=np.int32),
        "weights": np.array([], dtype=np.float32)
    })

    # Input 8: Size is 0 (all values ignored)
    list_of_inputs.append({
        "name": "bincount_8",
        "arr": np.array([1, 2, 3], dtype=np.int32),
        "size": np.array(0, dtype=np.int32),
        "weights": np.array([], dtype=np.float32)
    })

    # Input 9: Large size, empty int32 weights
    list_of_inputs.append({
        "name": "bincount_9",
        "arr": np.array([10, 20], dtype=np.int32),
        "size": np.array(30, dtype=np.int32),
        "weights": np.array([], dtype=np.int32)
    })

    # Input 10: int32 weights matching arr
    list_of_inputs.append({
        "name": "bincount_10",
        "arr": np.array([0, 2, 2, 1], dtype=np.int32),
        "size": np.array(3, dtype=np.int32),
        "weights": np.array([5, 10, 15, 20], dtype=np.int32)
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.Bincount"] = tf_raw_ops_bincount_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Cumsum_inputs():
    list_of_inputs = []

    # Input 1
    list_of_inputs.append({
        'exclusive': False,
        'reverse': False,
        'name': "cumsum_1d_float",
        'x': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        'axis': np.array(0, dtype=np.int32)
    })

    # Input 2
    list_of_inputs.append({
        'exclusive': True,
        'reverse': False,
        'name': "cumsum_1d_int_exclusive",
        'x': np.array([5, 10, 15], dtype=np.int32),
        'axis': np.array(0, dtype=np.int32)
    })

    # Input 3
    list_of_inputs.append({
        'exclusive': False,
        'reverse': True,
        'name': "cumsum_2d_double_reverse",
        'x': np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64),
        'axis': np.array(1, dtype=np.int32)
    })

    # Input 4
    list_of_inputs.append({
        'exclusive': True,
        'reverse': True,
        'name': "cumsum_2d_int64_both",
        'x': np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64),
        'axis': np.array(0, dtype=np.int64)
    })

    # Input 5
    list_of_inputs.append({
        'exclusive': False,
        'reverse': False,
        'name': "cumsum_3d_float",
        'x': np.array([[[1.0, -2.0], [3.0, -4.0]], [[5.0, -6.0], [7.0, -8.0]]], dtype=np.float32),
        'axis': np.array(2, dtype=np.int32)
    })

    # Input 6
    list_of_inputs.append({
        'exclusive': True,
        'reverse': False,
        'name': "cumsum_3d_int_axis_neg",
        'x': np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32),
        'axis': np.array(-1, dtype=np.int32)
    })

    # Input 7
    list_of_inputs.append({
        'exclusive': False,
        'reverse': True,
        'name': "cumsum_4d_float",
        'x': np.ones((2, 2, 2, 2), dtype=np.float32),
        'axis': np.array(1, dtype=np.int32)
    })

    # Input 8
    list_of_inputs.append({
        'exclusive': True,
        'reverse': True,
        'name': "cumsum_complex",
        'x': np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64),
        'axis': np.array(0, dtype=np.int32)
    })

    # Input 9
    list_of_inputs.append({
        'exclusive': False,
        'reverse': False,
        'name': "cumsum_int32_2",
        'x': np.array([[10, 20], [30, 40]], dtype=np.int32),
        'axis': np.array(-2, dtype=np.int64)
    })

    # Input 10
    list_of_inputs.append({
        'exclusive': False,
        'reverse': False,
        'name': "cumsum_double_1d",
        'x': np.array([0.1, 0.2, 0.3], dtype=np.float64),
        'axis': np.array(0, dtype=np.int32)
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.Cumsum"] = tf_raw_ops_Cumsum_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_EncodePng_inputs():
    list_of_inputs = []

    # Input 1: Grayscale (1 channel), uint8, default compression
    image_1 = np.random.randint(0, 256, size=(10, 10, 1), dtype=np.uint8)
    list_of_inputs.append({
        "image": image_1,
        "compression": -1,
        "name": None
    })

    # Input 2: RGB (3 channels), uint8, highest compression (9)
    image_2 = np.random.randint(0, 256, size=(20, 20, 3), dtype=np.uint8)
    list_of_inputs.append({
        "image": image_2,
        "compression": 9,
        "name": "encode_rgb_high"
    })

    # Input 3: RGBA (4 channels), uint8, no compression (0)
    image_3 = np.random.randint(0, 256, size=(15, 15, 4), dtype=np.uint8)
    list_of_inputs.append({
        "image": image_3,
        "compression": 0,
        "name": "encode_rgba_8bit_no_comp"
    })

    # Input 4: Grayscale + Alpha (2 channels), uint8, medium compression (5)
    image_4 = np.random.randint(0, 256, size=(32, 32, 2), dtype=np.uint8)
    list_of_inputs.append({
        "image": image_4,
        "compression": 5,
        "name": "gray_alpha"
    })

    # Input 5: Single pixel, RGB, uint8, default compression (-1)
    image_5 = np.random.randint(0, 256, size=(1, 1, 3), dtype=np.uint8)
    list_of_inputs.append({
        "image": image_5,
        "compression": -1,
        "name": "single_pixel_rgb"
    })

    # Input 6: Large image, Grayscale, uint8, fast compression (1)
    image_6 = np.random.randint(0, 256, size=(128, 128, 1), dtype=np.uint8)
    list_of_inputs.append({
        "image": image_6,
        "compression": 1,
        "name": "large_gray_fast"
    })

    # Input 7: RGBA, uint8, compression level 4
    image_7 = np.random.randint(0, 256, size=(50, 50, 4), dtype=np.uint8)
    list_of_inputs.append({
        "image": image_7,
        "compression": 4,
        "name": "rgba_mid"
    })

    # Input 8: RGB, uint8, compression level 7
    image_8 = np.random.randint(0, 256, size=(64, 48, 3), dtype=np.uint8)
    list_of_inputs.append({
        "image": image_8,
        "compression": 7,
        "name": "rgb_8bit_high"
    })

    # Input 9: Grayscale + Alpha, uint8, compression level 3
    image_9 = np.random.randint(0, 256, size=(16, 32, 2), dtype=np.uint8)
    list_of_inputs.append({
        "image": image_9,
        "compression": 3,
        "name": "gray_alpha_8bit"
    })

    # Input 10: Tall narrow image, RGB, uint8, compression level 8
    image_10 = np.random.randint(0, 256, size=(100, 10, 3), dtype=np.uint8)
    list_of_inputs.append({
        "image": image_10,
        "compression": 8,
        "name": "tall_narrow_rgb"
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.EncodePng"] = tf_raw_ops_EncodePng_inputs()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_FakeQuantWithMinMaxVarsGradient_inputs():
    list_of_inputs = []
    
    # Input 1
    input_dict_1 = {
        'num_bits': 8,
        'narrow_range': False,
        'name': "quant_grad_1",
        'gradients': np.array([1.0, 2.0], dtype=np.float32),
        'inputs': np.array([1.5, 2.5], dtype=np.float32),
        'min': np.array(0.0, dtype=np.float32),
        'max': np.array(6.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2
    input_dict_2 = {
        'num_bits': 4,
        'narrow_range': True,
        'name': "quant_grad_2",
        'gradients': np.array([[1.0, -1.0], [0.5, -0.5]], dtype=np.float32),
        'inputs': np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        'min': np.array(-1.0, dtype=np.float32),
        'max': np.array(1.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3
    input_dict_3 = {
        'num_bits': 2,
        'narrow_range': False,
        'name': "quant_grad_3",
        'gradients': np.array([[[1.0]]], dtype=np.float32),
        'inputs': np.array([[[0.5]]], dtype=np.float32),
        'min': np.array(-2.0, dtype=np.float32),
        'max': np.array(2.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4
    input_dict_4 = {
        'num_bits': 8,
        'narrow_range': False,
        'name': "quant_grad_4",
        'gradients': np.zeros((2, 3, 4), dtype=np.float32),
        'inputs': np.random.uniform(-5.0, 5.0, size=(2, 3, 4)).astype(np.float32),
        'min': np.array(-5.0, dtype=np.float32),
        'max': np.array(5.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5
    input_dict_5 = {
        'num_bits': 7,
        'narrow_range': True,
        'name': "quant_grad_5",
        'gradients': np.ones((5,), dtype=np.float32),
        'inputs': np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32),
        'min': np.array(-3.0, dtype=np.float32),
        'max': np.array(3.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6
    input_dict_6 = {
        'num_bits': 6,
        'narrow_range': False,
        'name': "quant_grad_6",
        'gradients': np.random.normal(size=(10,)).astype(np.float32),
        'inputs': np.random.normal(size=(10,)).astype(np.float32),
        'min': np.array(-1.5, dtype=np.float32),
        'max': np.array(1.5, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7
    input_dict_7 = {
        'num_bits': 3,
        'narrow_range': True,
        'name': "quant_grad_7",
        'gradients': np.array([0.0], dtype=np.float32),
        'inputs': np.array([0.0], dtype=np.float32),
        'min': np.array(-0.5, dtype=np.float32),
        'max': np.array(0.5, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8
    input_dict_8 = {
        'num_bits': 5,
        'narrow_range': False,
        'name': "quant_grad_8",
        'gradients': np.array([10.0, 20.0, 30.0], dtype=np.float32),
        'inputs': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'min': np.array(1.0, dtype=np.float32),
        'max': np.array(3.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9
    input_dict_9 = {
        'num_bits': 8,
        'narrow_range': True,
        'name': "quant_grad_9",
        'gradients': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        'inputs': np.array([[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]], dtype=np.float32),
        'min': np.array(1.0, dtype=np.float32),
        'max': np.array(2.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10
    input_dict_10 = {
        'num_bits': 8,
        'narrow_range': False,
        'name': "quant_grad_10",
        'gradients': np.random.uniform(-1.0, 1.0, size=(2, 2, 2, 2)).astype(np.float32),
        'inputs': np.random.uniform(-1.0, 1.0, size=(2, 2, 2, 2)).astype(np.float32),
        'min': np.array(-1.0, dtype=np.float32),
        'max': np.array(1.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.FakeQuantWithMinMaxVarsGradient"] = tf_raw_ops_FakeQuantWithMinMaxVarsGradient_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FusedBatchNormGradV3_inputs():
    list_of_inputs = []

    # Case 1: NHWC, float32, is_training=True
    input_dict = {
        'epsilon': 0.0001,
        'data_format': 'NHWC',
        'is_training': True,
        'name': 'fused_batch_norm_grad_1',
        'y_backprop': np.random.randn(2, 3, 4, 5).astype(np.float32),
        'x': np.random.randn(2, 3, 4, 5).astype(np.float32),
        'scale': np.random.randn(5).astype(np.float32),
        'reserve_space_1': np.random.randn(5).astype(np.float32),
        'reserve_space_2': np.abs(np.random.randn(5).astype(np.float32)) + 0.1,
        'reserve_space_3': np.random.randn(5).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: NHWC, float16, is_training=True
    input_dict = {
        'epsilon': 0.0001,
        'data_format': 'NHWC',
        'is_training': True,
        'name': 'fused_batch_norm_grad_2',
        'y_backprop': np.random.randn(1, 2, 2, 3).astype(np.float16),
        'x': np.random.randn(1, 2, 2, 3).astype(np.float16),
        'scale': np.random.randn(3).astype(np.float32),
        'reserve_space_1': np.random.randn(3).astype(np.float32),
        'reserve_space_2': np.abs(np.random.randn(3).astype(np.float32)) + 0.1,
        'reserve_space_3': np.random.randn(3).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: NCHW, float32, is_training=True
    input_dict = {
        'epsilon': 1e-4,
        'data_format': 'NCHW',
        'is_training': True,
        'name': 'fused_batch_norm_grad_3',
        'y_backprop': np.random.randn(2, 3, 2, 2).astype(np.float32),
        'x': np.random.randn(2, 3, 2, 2).astype(np.float32),
        'scale': np.random.randn(3).astype(np.float32),
        'reserve_space_1': np.random.randn(3).astype(np.float32),
        'reserve_space_2': np.abs(np.random.randn(3).astype(np.float32)) + 0.1,
        'reserve_space_3': np.random.randn(3).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: NHWC, float32, is_training=False, reserve_space_3 is empty
    input_dict = {
        'epsilon': 1e-4,
        'data_format': 'NHWC',
        'is_training': False,
        'name': 'fused_batch_norm_grad_4',
        'y_backprop': np.random.randn(2, 3, 4, 4).astype(np.float32),
        'x': np.random.randn(2, 3, 4, 4).astype(np.float32),
        'scale': np.random.randn(4).astype(np.float32),
        'reserve_space_1': np.random.randn(4).astype(np.float32),
        'reserve_space_2': np.abs(np.random.randn(4).astype(np.float32)) + 0.1,
        'reserve_space_3': np.array([], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: NHWC, float16, is_training=False, reserve_space_3 is empty
    input_dict = {
        'epsilon': 1e-5,
        'data_format': 'NHWC',
        'is_training': False,
        'name': 'fused_batch_norm_grad_5',
        'y_backprop': np.random.randn(1, 2, 2, 2).astype(np.float16),
        'x': np.random.randn(1, 2, 2, 2).astype(np.float16),
        'scale': np.random.randn(2).astype(np.float32),
        'reserve_space_1': np.random.randn(2).astype(np.float32),
        'reserve_space_2': np.abs(np.random.randn(2).astype(np.float32)) + 0.1,
        'reserve_space_3': np.array([], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: NHWC, float32, small size, is_training=True
    input_dict = {
        'epsilon': 1e-5,
        'data_format': 'NHWC',
        'is_training': True,
        'name': 'fused_batch_norm_grad_6',
        'y_backprop': np.random.randn(1, 1, 1, 1).astype(np.float32),
        'x': np.random.randn(1, 1, 1, 1).astype(np.float32),
        'scale': np.random.randn(1).astype(np.float32),
        'reserve_space_1': np.random.randn(1).astype(np.float32),
        'reserve_space_2': np.abs(np.random.randn(1).astype(np.float32)) + 0.1,
        'reserve_space_3': np.random.randn(1).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: NHWC, float32, negative inputs, is_training=True
    input_dict = {
        'epsilon': 1e-4,
        'data_format': 'NHWC',
        'is_training': True,
        'name': 'fused_batch_norm_grad_7',
        'y_backprop': -np.random.randn(2, 2, 2, 2).astype(np.float32),
        'x': -np.random.randn(2, 2, 2, 2).astype(np.float32),
        'scale': np.random.randn(2).astype(np.float32),
        'reserve_space_1': np.random.randn(2).astype(np.float32),
        'reserve_space_2': np.abs(np.random.randn(2).astype(np.float32)) + 0.1,
        'reserve_space_3': np.random.randn(2).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: NCHW, float32, larger scale, is_training=True
    input_dict = {
        'epsilon': 0.001,
        'data_format': 'NCHW',
        'is_training': True,
        'name': 'fused_batch_norm_grad_8',
        'y_backprop': np.random.randn(4, 8, 4, 4).astype(np.float32),
        'x': np.random.randn(4, 8, 4, 4).astype(np.float32),
        'scale': np.random.randn(8).astype(np.float32),
        'reserve_space_1': np.random.randn(8).astype(np.float32),
        'reserve_space_2': np.abs(np.random.randn(8).astype(np.float32)) + 0.1,
        'reserve_space_3': np.random.randn(8).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: NHWC, float16, larger scale, is_training=True
    input_dict = {
        'epsilon': 0.0001,
        'data_format': 'NHWC',
        'is_training': True,
        'name': 'fused_batch_norm_grad_9',
        'y_backprop': np.random.randn(3, 3, 3, 3).astype(np.float16),
        'x': np.random.randn(3, 3, 3, 3).astype(np.float16),
        'scale': np.random.randn(3).astype(np.float32),
        'reserve_space_1': np.random.randn(3).astype(np.float32),
        'reserve_space_2': np.abs(np.random.randn(3).astype(np.float32)) + 0.1,
        'reserve_space_3': np.random.randn(3).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: NHWC, float32, is_training=False, reserve_space_3 is empty
    input_dict = {
        'epsilon': 0.0001,
        'data_format': 'NHWC',
        'is_training': False,
        'name': 'fused_batch_norm_grad_10',
        'y_backprop': np.random.randn(3, 5, 5, 4).astype(np.float32),
        'x': np.random.randn(3, 5, 5, 4).astype(np.float32),
        'scale': np.random.randn(4).astype(np.float32),
        'reserve_space_1': np.random.randn(4).astype(np.float32),
        'reserve_space_2': np.abs(np.random.randn(4).astype(np.float32)) + 0.1,
        'reserve_space_3': np.array([], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.FusedBatchNormGradV3"] = tf_raw_ops_FusedBatchNormGradV3_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FusedBatchNormV2_inputs():
    list_of_inputs = []

    # Input 1: Training, NHWC, float32, basic
    input_dict = {
        "x": np.random.randn(2, 3, 3, 4).astype(np.float32),
        "scale": np.random.randn(4).astype(np.float32),
        "offset": np.random.randn(4).astype(np.float32),
        "mean": np.array([], dtype=np.float32),
        "variance": np.array([], dtype=np.float32),
        "epsilon": 0.0001,
        "exponential_avg_factor": 1.0,
        "data_format": "NHWC",
        "is_training": True,
        "name": "bn_train_nhwc_f32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Inference, NHWC, float32, basic
    input_dict = {
        "x": np.random.randn(2, 3, 3, 4).astype(np.float32),
        "scale": np.random.randn(4).astype(np.float32),
        "offset": np.random.randn(4).astype(np.float32),
        "mean": np.random.randn(4).astype(np.float32),
        "variance": np.abs(np.random.randn(4)).astype(np.float32),
        "epsilon": 0.001,
        "exponential_avg_factor": 1.0,
        "data_format": "NHWC",
        "is_training": False,
        "name": "bn_infer_nhwc_f32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Training, NCHW, float32
    input_dict = {
        "x": np.random.randn(2, 4, 3, 3).astype(np.float32),
        "scale": np.random.randn(4).astype(np.float32),
        "offset": np.random.randn(4).astype(np.float32),
        "mean": np.array([], dtype=np.float32),
        "variance": np.array([], dtype=np.float32),
        "epsilon": 0.0001,
        "exponential_avg_factor": 1.0,
        "data_format": "NCHW",
        "is_training": True,
        "name": "bn_train_nchw_f32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Inference, NCHW, float32
    input_dict = {
        "x": np.random.randn(2, 4, 3, 3).astype(np.float32),
        "scale": np.random.randn(4).astype(np.float32),
        "offset": np.random.randn(4).astype(np.float32),
        "mean": np.random.randn(4).astype(np.float32),
        "variance": np.abs(np.random.randn(4)).astype(np.float32),
        "epsilon": 1e-5,
        "exponential_avg_factor": 0.1,
        "data_format": "NCHW",
        "is_training": False,
        "name": "bn_infer_nchw_f32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Training, NHWC, float16
    input_dict = {
        "x": np.random.randn(1, 2, 2, 3).astype(np.float16),
        "scale": np.random.randn(3).astype(np.float32),
        "offset": np.random.randn(3).astype(np.float32),
        "mean": np.array([], dtype=np.float32),
        "variance": np.array([], dtype=np.float32),
        "epsilon": 0.0001,
        "exponential_avg_factor": 1.0,
        "data_format": "NHWC",
        "is_training": True,
        "name": "bn_train_nhwc_f16"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Inference, NHWC, float16
    input_dict = {
        "x": np.random.randn(1, 2, 2, 3).astype(np.float16),
        "scale": np.random.randn(3).astype(np.float32),
        "offset": np.random.randn(3).astype(np.float32),
        "mean": np.random.randn(3).astype(np.float32),
        "variance": np.abs(np.random.randn(3)).astype(np.float32),
        "epsilon": 1e-4,
        "exponential_avg_factor": 1.0,
        "data_format": "NHWC",
        "is_training": False,
        "name": "bn_infer_nhwc_f16"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Training, NHWC, larger batch
    input_dict = {
        "x": np.random.randn(8, 16, 16, 32).astype(np.float32),
        "scale": np.random.randn(32).astype(np.float32),
        "offset": np.random.randn(32).astype(np.float32),
        "mean": np.array([], dtype=np.float32),
        "variance": np.array([], dtype=np.float32),
        "epsilon": 1e-3,
        "exponential_avg_factor": 1.0,
        "data_format": "NHWC",
        "is_training": True,
        "name": "bn_train_large_nhwc"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Training, NCHW, larger batch, float16
    input_dict = {
        "x": np.random.randn(4, 16, 8, 8).astype(np.float16),
        "scale": np.random.randn(16).astype(np.float32),
        "offset": np.random.randn(16).astype(np.float32),
        "mean": np.array([], dtype=np.float32),
        "variance": np.array([], dtype=np.float32),
        "epsilon": 1e-4,
        "exponential_avg_factor": 1.0,
        "data_format": "NCHW",
        "is_training": True,
        "name": "bn_train_large_nchw_f16"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Inference, NHWC, zero variance
    input_dict = {
        "x": np.random.randn(2, 2, 2, 2).astype(np.float32),
        "scale": np.random.randn(2).astype(np.float32),
        "offset": np.random.randn(2).astype(np.float32),
        "mean": np.random.randn(2).astype(np.float32),
        "variance": np.zeros(2).astype(np.float32),
        "epsilon": 1e-5,
        "exponential_avg_factor": 1.0,
        "data_format": "NHWC",
        "is_training": False,
        "name": "bn_infer_zero_var"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Training, NCHW, minimal dimensions
    input_dict = {
        "x": np.random.randn(1, 1, 1, 1).astype(np.float32),
        "scale": np.random.randn(1).astype(np.float32),
        "offset": np.random.randn(1).astype(np.float32),
        "mean": np.array([], dtype=np.float32),
        "variance": np.array([], dtype=np.float32),
        "epsilon": 1e-4,
        "exponential_avg_factor": 1.0,
        "data_format": "NCHW",
        "is_training": True,
        "name": "bn_train_minimal"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.FusedBatchNormV2"] = tf_raw_ops_FusedBatchNormV2_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_FusedBatchNormV3_inputs():
    list_of_inputs = []

    # Input 1: NHWC, float32, is_training=True
    x = np.random.randn(2, 3, 3, 4).astype(np.float32)
    scale = np.random.randn(4).astype(np.float32)
    offset = np.random.randn(4).astype(np.float32)
    mean = np.array([], dtype=np.float32)
    variance = np.array([], dtype=np.float32)
    
    input_dict = {
        'epsilon': 0.0001,
        'exponential_avg_factor': 1.0,
        'data_format': 'NHWC',
        'is_training': True,
        'name': 'bn_case_1',
        'x': x,
        'scale': scale,
        'offset': offset,
        'mean': mean,
        'variance': variance
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: NCHW, float32, is_training=True
    x = np.random.randn(2, 4, 3, 3).astype(np.float32)
    scale = np.random.randn(4).astype(np.float32)
    offset = np.random.randn(4).astype(np.float32)
    mean = np.array([], dtype=np.float32)
    variance = np.array([], dtype=np.float32)
    
    input_dict = {
        'epsilon': 1e-5,
        'exponential_avg_factor': 1.0,
        'data_format': 'NCHW',
        'is_training': True,
        'name': 'bn_case_2',
        'x': x,
        'scale': scale,
        'offset': offset,
        'mean': mean,
        'variance': variance
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: NHWC, float32, is_training=False
    x = np.random.randn(2, 3, 3, 4).astype(np.float32)
    scale = np.random.randn(4).astype(np.float32)
    offset = np.random.randn(4).astype(np.float32)
    mean = np.random.randn(4).astype(np.float32)
    variance = np.abs(np.random.randn(4).astype(np.float32))
    
    input_dict = {
        'epsilon': 0.0001,
        'exponential_avg_factor': 1.0,
        'data_format': 'NHWC',
        'is_training': False,
        'name': 'bn_case_3',
        'x': x,
        'scale': scale,
        'offset': offset,
        'mean': mean,
        'variance': variance
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: NCHW, float32, is_training=False
    x = np.random.randn(2, 8, 4, 4).astype(np.float32)
    scale = np.random.randn(8).astype(np.float32)
    offset = np.random.randn(8).astype(np.float32)
    mean = np.random.randn(8).astype(np.float32)
    variance = np.abs(np.random.randn(8).astype(np.float32))
    
    input_dict = {
        'epsilon': 1e-4,
        'exponential_avg_factor': 1.0,
        'data_format': 'NCHW',
        'is_training': False,
        'name': 'bn_case_4',
        'x': x,
        'scale': scale,
        'offset': offset,
        'mean': mean,
        'variance': variance
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: NHWC, float16 (half) for x, is_training=True
    x = np.random.randn(4, 8, 8, 16).astype(np.float16)
    scale = np.random.randn(16).astype(np.float32)
    offset = np.random.randn(16).astype(np.float32)
    mean = np.array([], dtype=np.float32)
    variance = np.array([], dtype=np.float32)
    
    input_dict = {
        'epsilon': 0.0001,
        'exponential_avg_factor': 1.0,
        'data_format': 'NHWC',
        'is_training': True,
        'name': 'bn_case_5',
        'x': x,
        'scale': scale,
        'offset': offset,
        'mean': mean,
        'variance': variance
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: NCHW, float16 (half) for x, is_training=True
    x = np.random.randn(4, 16, 8, 8).astype(np.float16)
    scale = np.random.randn(16).astype(np.float32)
    offset = np.random.randn(16).astype(np.float32)
    mean = np.array([], dtype=np.float32)
    variance = np.array([], dtype=np.float32)
    
    input_dict = {
        'epsilon': 0.0001,
        'exponential_avg_factor': 1.0,
        'data_format': 'NCHW',
        'is_training': True,
        'name': 'bn_case_6',
        'x': x,
        'scale': scale,
        'offset': offset,
        'mean': mean,
        'variance': variance
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: NHWC, float16 (half) for x, is_training=False
    x = np.random.randn(4, 8, 8, 16).astype(np.float16)
    scale = np.random.randn(16).astype(np.float32)
    offset = np.random.randn(16).astype(np.float32)
    mean = np.random.randn(16).astype(np.float32)
    variance = np.abs(np.random.randn(16).astype(np.float32))
    
    input_dict = {
        'epsilon': 0.0001,
        'exponential_avg_factor': 1.0,
        'data_format': 'NHWC',
        'is_training': False,
        'name': 'bn_case_7',
        'x': x,
        'scale': scale,
        'offset': offset,
        'mean': mean,
        'variance': variance
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: NCHW, float16 (half) for x, is_training=False
    x = np.random.randn(4, 16, 8, 8).astype(np.float16)
    scale = np.random.randn(16).astype(np.float32)
    offset = np.random.randn(16).astype(np.float32)
    mean = np.random.randn(16).astype(np.float32)
    variance = np.abs(np.random.randn(16).astype(np.float32))
    
    input_dict = {
        'epsilon': 0.0001,
        'exponential_avg_factor': 1.0,
        'data_format': 'NCHW',
        'is_training': False,
        'name': 'bn_case_8',
        'x': x,
        'scale': scale,
        'offset': offset,
        'mean': mean,
        'variance': variance
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: NHWC, large batch, float32, is_training=True
    x = np.random.randn(32, 16, 16, 32).astype(np.float32)
    scale = np.random.randn(32).astype(np.float32)
    offset = np.random.randn(32).astype(np.float32)
    mean = np.array([], dtype=np.float32)
    variance = np.array([], dtype=np.float32)
    
    input_dict = {
        'epsilon': 1e-5,
        'exponential_avg_factor': 1.0,
        'data_format': 'NHWC',
        'is_training': True,
        'name': 'bn_case_9',
        'x': x,
        'scale': scale,
        'offset': offset,
        'mean': mean,
        'variance': variance
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: NHWC, large batch, float32, is_training=False
    x = np.random.randn(32, 16, 16, 32).astype(np.float32)
    scale = np.random.randn(32).astype(np.float32)
    offset = np.random.randn(32).astype(np.float32)
    mean = np.random.randn(32).astype(np.float32)
    variance = np.abs(np.random.randn(32).astype(np.float32))
    
    input_dict = {
        'epsilon': 1e-5,
        'exponential_avg_factor': 1.0,
        'data_format': 'NHWC',
        'is_training': False,
        'name': 'bn_case_10',
        'x': x,
        'scale': scale,
        'offset': offset,
        'mean': mean,
        'variance': variance
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.FusedBatchNormV3"] = tf_raw_ops_FusedBatchNormV3_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MatrixDiagPartV2_inputs():
    list_of_inputs = []

    # Input 1: float32 matrix with single diagonal (k = 0)
    list_of_inputs.append({
        "name": "diag_part_1",
        "input": np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], dtype=np.float32),
        "k": np.array(0, dtype=np.int32),
        "padding_value": np.array(0.0, dtype=np.float32)
    })

    # Input 2: int32 matrix with a single negative diagonal (k = -1)
    list_of_inputs.append({
        "name": "diag_part_2",
        "input": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32),
        "k": np.array(-1, dtype=np.int32),
        "padding_value": np.array(-1, dtype=np.int32)
    })

    # Input 3: float64 matrix with a band of diagonals (k = [-1, 1])
    list_of_inputs.append({
        "name": "diag_part_3",
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64),
        "k": np.array([-1, 1], dtype=np.int32),
        "padding_value": np.array(9.0, dtype=np.float64)
    })

    # Input 4: 3D float32 tensor with a superdiagonal band (k = [1, 3])
    list_of_inputs.append({
        "name": "diag_part_4",
        "input": np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6]], [[5, 4, 3, 2], [1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.float32),
        "k": np.array([1, 3], dtype=np.int32),
        "padding_value": np.array(9.0, dtype=np.float32)
    })

    # Input 5: int64 matrix with a single negative diagonal (k = -1)
    list_of_inputs.append({
        "name": "diag_part_5",
        "input": np.arange(10, dtype=np.int64).reshape((2, 5)),
        "k": np.array(-1, dtype=np.int32),
        "padding_value": np.array(0, dtype=np.int64)
    })

    # Input 6: complex64 4D tensor with a diagonal band (k = [0, 1])
    list_of_inputs.append({
        "name": "diag_part_6",
        "input": np.arange(36, dtype=np.complex64).reshape((2, 2, 3, 3)),
        "k": np.array([0, 1], dtype=np.int32),
        "padding_value": np.array(0 + 0j, dtype=np.complex64)
    })

    # Input 7: uint8 matrix with a subdiagonal band (k = [-2, -1])
    list_of_inputs.append({
        "name": "diag_part_7",
        "input": np.ones((4, 4), dtype=np.uint8),
        "k": np.array([-2, -1], dtype=np.int32),
        "padding_value": np.array(255, dtype=np.uint8)
    })

    # Input 8: float32 3D tensor, main diagonal (k = 0)
    list_of_inputs.append({
        "name": "diag_part_8",
        "input": np.zeros((3, 1, 2), dtype=np.float32),
        "k": np.array(0, dtype=np.int32),
        "padding_value": np.array(-99.9, dtype=np.float32)
    })

    # Input 9: int32 matrix with a wider subdiagonal band (k = [-2, 0])
    list_of_inputs.append({
        "name": "diag_part_9",
        "input": np.arange(8, dtype=np.int32).reshape((4, 2)),
        "k": np.array([-2, 0], dtype=np.int32),
        "padding_value": np.array(999, dtype=np.int32)
    })

    # Input 10: int16 matrix with superdiagonals (k = [1, 2])
    list_of_inputs.append({
        "name": "diag_part_10",
        "input": np.arange(25, dtype=np.int16).reshape((5, 5)),
        "k": np.array([1, 2], dtype=np.int32),
        "padding_value": np.array(42, dtype=np.int16)
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.MatrixDiagPartV2"] = tf_raw_ops_MatrixDiagPartV2_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MatrixDiagPartV3_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "align": "RIGHT_LEFT",
        "name": "diag_1",
        "input": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        "k": np.array(0, dtype=np.int32),
        "padding_value": np.array(0, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "align": "LEFT_RIGHT",
        "name": "diag_2",
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "k": np.array([-1, 1], dtype=np.int32),
        "padding_value": np.array(-1.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "align": "LEFT_LEFT",
        "name": "diag_3",
        "input": np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int32),
        "k": np.array(-1, dtype=np.int32),
        "padding_value": np.array(9, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "align": "RIGHT_RIGHT",
        "name": "diag_4",
        "input": np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.int64),
        "k": np.array([1, 2], dtype=np.int32),
        "padding_value": np.array(0, dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "align": "LEFT_RIGHT",
        "name": "diag_5",
        "input": np.array([[1.5, 2.5, 3.5]], dtype=np.float64),
        "k": np.array(0, dtype=np.int32),
        "padding_value": np.array(0.0, dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "align": "RIGHT_LEFT",
        "name": "diag_6",
        "input": np.zeros((2, 2, 3, 4), dtype=np.float32),
        "k": np.array([-2, 1], dtype=np.int32),
        "padding_value": np.array(99.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "align": "LEFT_RIGHT",
        "name": "diag_7",
        "input": np.array([[True, False], [False, True]], dtype=bool),
        "k": np.array(0, dtype=np.int32),
        "padding_value": np.array(False, dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "align": "RIGHT_LEFT",
        "name": "diag_8",
        "input": np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64),
        "k": np.array([0, 1], dtype=np.int32),
        "padding_value": np.array(0+0j, dtype=np.complex64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "align": "LEFT_RIGHT",
        "name": "diag_9",
        "input": np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.int32),
        "k": np.array([-2, -1], dtype=np.int32),
        "padding_value": np.array(-99, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "align": "RIGHT_RIGHT",
        "name": "diag_10",
        "input": np.ones((3, 4, 4), dtype=np.int32),
        "k": np.array([-1, 2], dtype=np.int32),
        "padding_value": np.array(-1, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.MatrixDiagPartV3"] = tf_raw_ops_MatrixDiagPartV3_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Mean_inputs():
    list_of_inputs = []
    
    # Input 1
    list_of_inputs.append({
        'keep_dims': False,
        'name': 'mean_1',
        'input': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'axis': np.array(0, dtype=np.int32)
    })
    
    # Input 2
    list_of_inputs.append({
        'keep_dims': True,
        'name': 'mean_2',
        'input': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        'axis': np.array([0], dtype=np.int32)
    })
    
    # Input 3
    list_of_inputs.append({
        'keep_dims': False,
        'name': 'mean_3',
        'input': np.array([[1, 2], [3, 4]], dtype=np.int32),
        'axis': np.array([1], dtype=np.int64)
    })
    
    # Input 4
    list_of_inputs.append({
        'keep_dims': True,
        'name': 'mean_4',
        'input': np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32),
        'axis': np.array([0, 2], dtype=np.int32)
    })
    
    # Input 5
    list_of_inputs.append({
        'keep_dims': False,
        'name': 'mean_5',
        'input': np.array([10, 20, 30], dtype=np.int64),
        'axis': np.array([-1], dtype=np.int32)
    })
    
    # Input 6
    list_of_inputs.append({
        'keep_dims': False,
        'name': 'mean_6',
        'input': np.array([1.5, 3.5], dtype=np.float32),
        'axis': np.array(0, dtype=np.int32)
    })
    
    # Input 7
    list_of_inputs.append({
        'keep_dims': True,
        'name': 'mean_7',
        'input': np.array([[1, 2, 3]], dtype=np.int64),
        'axis': np.array([1], dtype=np.int32)
    })
    
    # Input 8
    list_of_inputs.append({
        'keep_dims': False,
        'name': 'mean_8',
        'input': np.array([1.0, 2.0], dtype=np.float32),
        'axis': np.array(0, dtype=np.int64)
    })
    
    # Input 9
    list_of_inputs.append({
        'keep_dims': False,
        'name': 'mean_9',
        'input': np.array([[1, 2], [3, 4]], dtype=np.int32),
        'axis': np.array([0, 1], dtype=np.int32)
    })
    
    # Input 10
    list_of_inputs.append({
        'keep_dims': True,
        'name': 'mean_10',
        'input': np.array([10.0, 20.0], dtype=np.float64),
        'axis': np.array([], dtype=np.int32)
    })
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Mean"] = tf_raw_ops_Mean_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_NonMaxSuppression_inputs():
    list_of_inputs = []

    # Input 1: Basic standard input
    boxes1 = np.array([[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9]], dtype=np.float32)
    scores1 = np.array([0.9, 0.75, 0.6], dtype=np.float32)
    max_output_size1 = np.array(2, dtype=np.int32)
    iou_threshold1 = 0.5
    name1 = "nms_1"
    list_of_inputs.append({
        'iou_threshold': iou_threshold1,
        'name': name1,
        'boxes': boxes1,
        'scores': scores1,
        'max_output_size': max_output_size1
    })

    # Input 2: Zero overlapping boxes
    boxes2 = np.array([[0, 0, 1, 1], [2, 2, 3, 3]], dtype=np.float32)
    scores2 = np.array([0.5, 0.8], dtype=np.float32)
    max_output_size2 = np.array(5, dtype=np.int32)
    iou_threshold2 = 0.3
    name2 = "nms_2"
    list_of_inputs.append({
        'iou_threshold': iou_threshold2,
        'name': name2,
        'boxes': boxes2,
        'scores': scores2,
        'max_output_size': max_output_size2
    })

    # Input 3: Negative/large coordinates
    boxes3 = np.array([[-10, -10, 10, 10], [-5, -5, 5, 5], [0, 0, 15, 15]], dtype=np.float32)
    scores3 = np.array([-0.1, 0.9, 0.4], dtype=np.float32)
    max_output_size3 = np.array(1, dtype=np.int32)
    iou_threshold3 = 0.1
    name3 = "nms_3"
    list_of_inputs.append({
        'iou_threshold': iou_threshold3,
        'name': name3,
        'boxes': boxes3,
        'scores': scores3,
        'max_output_size': max_output_size3
    })

    # Input 4: Empty boxes
    boxes4 = np.zeros((0, 4), dtype=np.float32)
    scores4 = np.zeros((0,), dtype=np.float32)
    max_output_size4 = np.array(3, dtype=np.int32)
    iou_threshold4 = 0.5
    name4 = "nms_4"
    list_of_inputs.append({
        'iou_threshold': iou_threshold4,
        'name': name4,
        'boxes': boxes4,
        'scores': scores4,
        'max_output_size': max_output_size4
    })

    # Input 5: Max output size is 0
    boxes5 = np.array([[0, 0, 1, 1], [0.5, 0.5, 1.5, 1.5]], dtype=np.float32)
    scores5 = np.array([0.9, 0.8], dtype=np.float32)
    max_output_size5 = np.array(0, dtype=np.int32)
    iou_threshold5 = 0.5
    name5 = "nms_5"
    list_of_inputs.append({
        'iou_threshold': iou_threshold5,
        'name': name5,
        'boxes': boxes5,
        'scores': scores5,
        'max_output_size': max_output_size5
    })

    # Input 6: Highly overlapping boxes, low iou threshold
    boxes6 = np.array([[0, 0, 10, 10], [1, 1, 9, 9], [2, 2, 8, 8]], dtype=np.float32)
    scores6 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    max_output_size6 = np.array(10, dtype=np.int32)
    iou_threshold6 = 0.05
    name6 = "nms_6"
    list_of_inputs.append({
        'iou_threshold': iou_threshold6,
        'name': name6,
        'boxes': boxes6,
        'scores': scores6,
        'max_output_size': max_output_size6
    })

    # Input 7: iou_threshold is 1.0 (no pruning)
    boxes7 = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32)
    scores7 = np.array([0.5, 0.5], dtype=np.float32)
    max_output_size7 = np.array(2, dtype=np.int32)
    iou_threshold7 = 1.0
    name7 = "nms_7"
    list_of_inputs.append({
        'iou_threshold': iou_threshold7,
        'name': name7,
        'boxes': boxes7,
        'scores': scores7,
        'max_output_size': max_output_size7
    })

    # Input 8: Many boxes with random coordinates
    np.random.seed(42)
    boxes8 = np.random.rand(100, 4).astype(np.float32)
    for i in range(100):
        if boxes8[i, 2] < boxes8[i, 0]:
            boxes8[i, 0], boxes8[i, 2] = boxes8[i, 2], boxes8[i, 0]
        if boxes8[i, 3] < boxes8[i, 1]:
            boxes8[i, 1], boxes8[i, 3] = boxes8[i, 3], boxes8[i, 1]
    scores8 = np.random.rand(100).astype(np.float32)
    max_output_size8 = np.array(15, dtype=np.int32)
    iou_threshold8 = 0.4
    name8 = "nms_8"
    list_of_inputs.append({
        'iou_threshold': iou_threshold8,
        'name': name8,
        'boxes': boxes8,
        'scores': scores8,
        'max_output_size': max_output_size8
    })

    # Input 9: Identical boxes, varying scores
    boxes9 = np.array([[0, 0, 1, 1]] * 10, dtype=np.float32)
    scores9 = np.array([float(i) for i in range(10)], dtype=np.float32)
    max_output_size9 = np.array(5, dtype=np.int32)
    iou_threshold9 = 0.5
    name9 = "nms_9"
    list_of_inputs.append({
        'iou_threshold': iou_threshold9,
        'name': name9,
        'boxes': boxes9,
        'scores': scores9,
        'max_output_size': max_output_size9
    })

    # Input 10: Non-overlapping sequential boxes
    boxes10 = np.array([[i, i, i + 0.5, i + 0.5] for i in range(10)], dtype=np.float32)
    scores10 = np.array([0.1 * i for i in range(10)], dtype=np.float32)
    max_output_size10 = np.array(100, dtype=np.int32)
    iou_threshold10 = 0.5
    name10 = "nms_10"
    list_of_inputs.append({
        'iou_threshold': iou_threshold10,
        'name': name10,
        'boxes': boxes10,
        'scores': scores10,
        'max_output_size': max_output_size10
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.NonMaxSuppression"] = tf_raw_ops_NonMaxSuppression_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_NthElement_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'reverse': False,
        'name': "nth_element_1",
        'input': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'n': np.array(1, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'reverse': True,
        'name': "nth_element_2",
        'input': np.array([[5, 2, 9, 1], [3, 8, 4, 7]], dtype=np.int32),
        'n': np.array(2, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'reverse': False,
        'name': "nth_element_3",
        'input': np.array([[[1.5, -2.3], [0.0, 4.1]], [[-1.1, 2.2], [3.3, -4.4]]], dtype=np.float64),
        'n': np.array(0, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'reverse': False,
        'name': "nth_element_4",
        'input': np.array([10, 20, 30, 40, 50], dtype=np.int64),
        'n': np.array(4, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'reverse': True,
        'name': "nth_element_5",
        'input': np.array([1.1, 2.2, 3.3], dtype=np.float32),
        'n': np.array(1, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'reverse': False,
        'name': "nth_element_6",
        'input': np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32),
        'n': np.array(0, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'reverse': True,
        'name': "nth_element_7",
        'input': np.array([[-1, -2, -3, -4], [5, 6, 7, 8]], dtype=np.int64),
        'n': np.array(3, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'reverse': False,
        'name': "nth_element_8",
        'input': np.array([3.14, 2.71, 1.41], dtype=np.float64),
        'n': np.array(2, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'reverse': True,
        'name': "nth_element_9",
        'input': np.array([[[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]]], dtype=np.float32),
        'n': np.array(1, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'reverse': False,
        'name': "nth_element_10",
        'input': np.array([1000, 2000, 3000, 4000], dtype=np.int32),
        'n': np.array(2, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.NthElement"] = tf_raw_ops_NthElement_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ParameterizedTruncatedNormal_inputs():
    list_of_inputs = []

    # Input 1: Float32, 1D shape, all scalar parameters
    shape_1 = np.array([5], dtype=np.int32)
    means_1 = np.array(0.0, dtype=np.float32)
    stdevs_1 = np.array(1.0, dtype=np.float32)
    minvals_1 = np.array(-2.0, dtype=np.float32)
    maxvals_1 = np.array(2.0, dtype=np.float32)
    input_dict_1 = {
        'seed': 42,
        'seed2': 1,
        'name': "truncated_1",
        'shape': shape_1,
        'means': means_1,
        'stdevs': stdevs_1,
        'minvals': minvals_1,
        'maxvals': maxvals_1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Float32, 2D shape, all vector parameters matching batch size 3
    shape_2 = np.array([3, 4], dtype=np.int32)
    means_2 = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    stdevs_2 = np.array([1.0, 0.5, 2.0], dtype=np.float32)
    minvals_2 = np.array([-1.0, 0.0, -3.0], dtype=np.float32)
    maxvals_2 = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    input_dict_2 = {
        'seed': 43,
        'seed2': 123,
        'name': "truncated_2",
        'shape': shape_2,
        'means': means_2,
        'stdevs': stdevs_2,
        'minvals': minvals_2,
        'maxvals': maxvals_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Float64, 3D shape, all scalar parameters, int64 shape
    shape_3 = np.array([2, 2, 2], dtype=np.int64)
    means_3 = np.array(5.0, dtype=np.float64)
    stdevs_3 = np.array(0.1, dtype=np.float64)
    minvals_3 = np.array(4.5, dtype=np.float64)
    maxvals_3 = np.array(5.5, dtype=np.float64)
    input_dict_3 = {
        'seed': 1,
        'seed2': 2,
        'name': "truncated_3",
        'shape': shape_3,
        'means': means_3,
        'stdevs': stdevs_3,
        'minvals': minvals_3,
        'maxvals': maxvals_3
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Float16, 2D shape, all vector parameters matching batch size 4
    shape_4 = np.array([4, 1], dtype=np.int32)
    means_4 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float16)
    stdevs_4 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float16)
    minvals_4 = np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float16)
    maxvals_4 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    input_dict_4 = {
        'seed': 44,
        'seed2': 12,
        'name': "truncated_4",
        'shape': shape_4,
        'means': means_4,
        'stdevs': stdevs_4,
        'minvals': minvals_4,
        'maxvals': maxvals_4
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Float32, 2D shape, all vector parameters matching batch size 2
    shape_5 = np.array([2, 3], dtype=np.int32)
    means_5 = np.array([-1.0, 1.0], dtype=np.float32)
    stdevs_5 = np.array([1.0, 1.0], dtype=np.float32)
    minvals_5 = np.array([-5.0, -5.0], dtype=np.float32)
    maxvals_5 = np.array([0.0, 5.0], dtype=np.float32)
    input_dict_5 = {
        'seed': 7,
        'seed2': 8,
        'name': "truncated_5",
        'shape': shape_5,
        'means': means_5,
        'stdevs': stdevs_5,
        'minvals': minvals_5,
        'maxvals': maxvals_5
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Float32, 1D shape, large finite bounds
    shape_6 = np.array([3], dtype=np.int32)
    means_6 = np.array(0.0, dtype=np.float32)
    stdevs_6 = np.array(1.0, dtype=np.float32)
    minvals_6 = np.array(-1000.0, dtype=np.float32)
    maxvals_6 = np.array(1000.0, dtype=np.float32)
    input_dict_6 = {
        'seed': 10,
        'seed2': 20,
        'name': "truncated_6",
        'shape': shape_6,
        'means': means_6,
        'stdevs': stdevs_6,
        'minvals': minvals_6,
        'maxvals': maxvals_6
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Float32, 4D shape (int64), all scalar parameters
    shape_7 = np.array([2, 2, 2, 2], dtype=np.int64)
    means_7 = np.array(10.0, dtype=np.float32)
    stdevs_7 = np.array(5.0, dtype=np.float32)
    minvals_7 = np.array(0.0, dtype=np.float32)
    maxvals_7 = np.array(20.0, dtype=np.float32)
    input_dict_7 = {
        'seed': 99,
        'seed2': 99,
        'name': "truncated_7",
        'shape': shape_7,
        'means': means_7,
        'stdevs': stdevs_7,
        'minvals': minvals_7,
        'maxvals': maxvals_7
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Float64, 1D shape, all vector parameters matching batch size 5
    shape_8 = np.array([5], dtype=np.int32)
    means_8 = np.array([1., 2., 3., 4., 5.], dtype=np.float64)
    stdevs_8 = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    minvals_8 = np.array([0., 0., 0., 0., 0.], dtype=np.float64)
    maxvals_8 = np.array([10., 10., 10., 10., 10.], dtype=np.float64)
    input_dict_8 = {
        'seed': 42,
        'seed2': 42,
        'name': "truncated_8",
        'shape': shape_8,
        'means': means_8,
        'stdevs': stdevs_8,
        'minvals': minvals_8,
        'maxvals': maxvals_8
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Float32, 1D shape, reasonable standard deviation, scalar parameters
    shape_9 = np.array([2], dtype=np.int32)
    means_9 = np.array(0.0, dtype=np.float32)
    stdevs_9 = np.array(0.5, dtype=np.float32)
    minvals_9 = np.array(-1.0, dtype=np.float32)
    maxvals_9 = np.array(1.0, dtype=np.float32)
    input_dict_9 = {
        'seed': 123,
        'seed2': 456,
        'name': "truncated_9",
        'shape': shape_9,
        'means': means_9,
        'stdevs': stdevs_9,
        'minvals': minvals_9,
        'maxvals': maxvals_9
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Float32, 3D shape, all vector parameters matching batch size 4
    shape_10 = np.array([4, 2, 2], dtype=np.int32)
    means_10 = np.array([-10.0, 0.0, 10.0, 20.0], dtype=np.float32)
    stdevs_10 = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    minvals_10 = np.array([-100.0, -100.0, -100.0, -100.0], dtype=np.float32)
    maxvals_10 = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float32)
    input_dict_10 = {
        'seed': 111,
        'seed2': 222,
        'name': "truncated_10",
        'shape': shape_10,
        'means': means_10,
        'stdevs': stdevs_10,
        'minvals': minvals_10,
        'maxvals': maxvals_10
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.ParameterizedTruncatedNormal"] = tf_raw_ops_ParameterizedTruncatedNormal_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseAddGrad_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'name': "sparse_add_grad_1",
        'backprop_val_grad': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'a_indices': np.array([[0, 0], [1, 1]], dtype=np.int64),
        'b_indices': np.array([[0, 0], [1, 2]], dtype=np.int64),
        'sum_indices': np.array([[0, 0], [1, 1], [1, 2]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'name': "sparse_add_grad_2",
        'backprop_val_grad': np.array([0.5, -1.5, 2.5], dtype=np.float32),
        'a_indices': np.array([[1], [3]], dtype=np.int64),
        'b_indices': np.array([[2], [3]], dtype=np.int64),
        'sum_indices': np.array([[1], [2], [3]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'name': "sparse_add_grad_3",
        'backprop_val_grad': np.array([10, 20], dtype=np.int32),
        'a_indices': np.array([[0, 1]], dtype=np.int64),
        'b_indices': np.array([[1, 0]], dtype=np.int64),
        'sum_indices': np.array([[0, 1], [1, 0]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'name': "sparse_add_grad_4",
        'backprop_val_grad': np.array([], dtype=np.float64),
        'a_indices': np.empty((0, 2), dtype=np.int64),
        'b_indices': np.empty((0, 2), dtype=np.int64),
        'sum_indices': np.empty((0, 2), dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'name': "sparse_add_grad_5",
        'backprop_val_grad': np.array([1.0, 2.0, 3.0], dtype=np.float64),
        'a_indices': np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int64),
        'b_indices': np.array([[0, 1, 0], [1, 0, 0]], dtype=np.int64),
        'sum_indices': np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'name': "sparse_add_grad_6",
        'backprop_val_grad': np.array([1.0 + 2.0j, 2.0 - 1.0j], dtype=np.complex64),
        'a_indices': np.array([[0, 0], [0, 1]], dtype=np.int64),
        'b_indices': np.array([[0, 0], [0, 1]], dtype=np.int64),
        'sum_indices': np.array([[0, 0], [0, 1]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'name': "sparse_add_grad_7",
        'backprop_val_grad': np.array([1, 2, 3], dtype=np.int64),
        'a_indices': np.array([[10, 20], [30, 40]], dtype=np.int64),
        'b_indices': np.array([[15, 25]], dtype=np.int64),
        'sum_indices': np.array([[10, 20], [15, 25], [30, 40]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'name': "sparse_add_grad_8",
        'backprop_val_grad': np.array([4], dtype=np.int8),
        'a_indices': np.array([[0]], dtype=np.int64),
        'b_indices': np.array([[0]], dtype=np.int64),
        'sum_indices': np.array([[0]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'name': "sparse_add_grad_9",
        'backprop_val_grad': np.array([1.0, 2.0], dtype=np.float32),
        'a_indices': np.array([[0, 0, 0, 0]], dtype=np.int64),
        'b_indices': np.array([[0, 0, 0, 1]], dtype=np.int64),
        'sum_indices': np.array([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'name': "sparse_add_grad_10",
        'backprop_val_grad': np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        'a_indices': np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64),
        'b_indices': np.array([[0, 1], [1, 0], [1, 1]], dtype=np.int64),
        'sum_indices': np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseAddGrad"] = tf_raw_ops_SparseAddGrad_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sparse_reduce_max_inputs():
    list_of_inputs = []

    # Case 1
    input_dict_1 = {
        'keep_dims': False,
        'name': "op1",
        'input_indices': np.array([[0, 0], [1, 2]], dtype=np.int64),
        'input_values': np.array([1.5, 2.5], dtype=np.float32),
        'input_shape': np.array([3, 4], dtype=np.int64),
        'reduction_axes': np.array([0], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2
    input_dict_2 = {
        'keep_dims': True,
        'name': "op2",
        'input_indices': np.array([[0, 0], [1, 1]], dtype=np.int64),
        'input_values': np.array([-1, -2], dtype=np.int32),
        'input_shape': np.array([2, 2], dtype=np.int64),
        'reduction_axes': np.array([1], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3
    input_dict_3 = {
        'keep_dims': False,
        'name': "op3",
        'input_indices': np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64),
        'input_values': np.array([5.0, 10.0], dtype=np.float64),
        'input_shape': np.array([2, 2, 2], dtype=np.int64),
        'reduction_axes': np.array([0, 2], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4
    input_dict_4 = {
        'keep_dims': True,
        'name': "op4",
        'input_indices': np.array([[0], [1], [2]], dtype=np.int64),
        'input_values': np.array([10, 20, 30], dtype=np.int64),
        'input_shape': np.array([5], dtype=np.int64),
        'reduction_axes': np.array([0], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5
    input_dict_5 = {
        'keep_dims': False,
        'name': "op5",
        'input_indices': np.array([[0, 1]], dtype=np.int64),
        'input_values': np.array([42], dtype=np.int32),
        'input_shape': np.array([3, 3], dtype=np.int64),
        'reduction_axes': np.array([0, 1], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6
    input_dict_6 = {
        'keep_dims': False,
        'name': "op6",
        'input_indices': np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64),
        'input_values': np.array([1, 2, 3], dtype=np.uint8),
        'input_shape': np.array([2, 2], dtype=np.int64),
        'reduction_axes': np.array([-1], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7
    input_dict_7 = {
        'keep_dims': True,
        'name': "op7",
        'input_indices': np.array([[0, 0, 0]], dtype=np.int64),
        'input_values': np.array([0.5], dtype=np.float32),
        'input_shape': np.array([1, 1, 1], dtype=np.int64),
        'reduction_axes': np.array([-1], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8
    input_dict_8 = {
        'keep_dims': False,
        'name': "op8",
        'input_indices': np.array([[0, 1, 2, 3]], dtype=np.int64),
        'input_values': np.array([100], dtype=np.int16),
        'input_shape': np.array([5, 5, 5, 5], dtype=np.int64),
        'reduction_axes': np.array([1, 3], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9
    input_dict_9 = {
        'keep_dims': False,
        'name': "op9",
        'input_indices': np.array([[0, 0], [1, 1]], dtype=np.int64),
        'input_values': np.array([1, 2], dtype=np.int8),
        'input_shape': np.array([5, 5], dtype=np.int64),
        'reduction_axes': np.array([0], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10
    input_dict_10 = {
        'keep_dims': True,
        'name': "op10",
        'input_indices': np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int64),
        'input_values': np.array([1.23, 4.56], dtype=np.float64),
        'input_shape': np.array([3, 3, 3], dtype=np.int64),
        'reduction_axes': np.array([1], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseReduceMax"] = tf_raw_ops_sparse_reduce_max_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseReduceSum_inputs():
    list_of_inputs = []
    
    # Input 1
    list_of_inputs.append({
        'keep_dims': False,
        'name': "test1",
        'input_indices': np.array([[0, 0], [1, 2]], dtype=np.int64),
        'input_values': np.array([1.5, 2.5], dtype=np.float32),
        'input_shape': np.array([2, 3], dtype=np.int64),
        'reduction_axes': np.array([0], dtype=np.int32)
    })
    
    # Input 2
    list_of_inputs.append({
        'keep_dims': True,
        'name': "test2",
        'input_indices': np.array([[0, 0], [1, 2]], dtype=np.int64),
        'input_values': np.array([1.5, 2.5], dtype=np.float32),
        'input_shape': np.array([2, 3], dtype=np.int64),
        'reduction_axes': np.array([1], dtype=np.int32)
    })
    
    # Input 3
    list_of_inputs.append({
        'keep_dims': False,
        'name': "test3",
        'input_indices': np.array([[0, 0], [1, 1]], dtype=np.int64),
        'input_values': np.array([1.0, 2.0], dtype=np.float32),
        'input_shape': np.array([2, 2], dtype=np.int64),
        'reduction_axes': np.array([0, 1], dtype=np.int32)
    })
    
    # Input 4
    list_of_inputs.append({
        'keep_dims': False,
        'name': "test4",
        'input_indices': np.array([[0, 0], [1, 2]], dtype=np.int64),
        'input_values': np.array([1.5, 2.5], dtype=np.float32),
        'input_shape': np.array([2, 3], dtype=np.int64),
        'reduction_axes': np.array([-1], dtype=np.int32)
    })
    
    # Input 5
    list_of_inputs.append({
        'keep_dims': False,
        'name': "test5",
        'input_indices': np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64),
        'input_values': np.array([10, 20], dtype=np.int32),
        'input_shape': np.array([2, 2, 2], dtype=np.int64),
        'reduction_axes': np.array([0, 2], dtype=np.int32)
    })
    
    # Input 6
    list_of_inputs.append({
        'keep_dims': True,
        'name': "test6",
        'input_indices': np.array([[0], [2], [4]], dtype=np.int64),
        'input_values': np.array([1.0, 2.0, 3.0], dtype=np.float64),
        'input_shape': np.array([5], dtype=np.int64),
        'reduction_axes': np.array([0], dtype=np.int32)
    })
    
    # Input 7
    list_of_inputs.append({
        'keep_dims': False,
        'name': "test7",
        'input_indices': np.array([[0, 0], [1, 1]], dtype=np.int64),
        'input_values': np.array([1.0, 2.0], dtype=np.float32),
        'input_shape': np.array([2, 2], dtype=np.int64),
        'reduction_axes': np.array([], dtype=np.int32)
    })
    
    # Input 8
    list_of_inputs.append({
        'keep_dims': True,
        'name': "test8",
        'input_indices': np.array([[0, 1, 2, 3]], dtype=np.int64),
        'input_values': np.array([5.0], dtype=np.float32),
        'input_shape': np.array([2, 3, 4, 5], dtype=np.int64),
        'reduction_axes': np.array([1, 3], dtype=np.int32)
    })
    
    # Input 9
    list_of_inputs.append({
        'keep_dims': False,
        'name': "test9",
        'input_indices': np.array([[0, 0], [1, 1]], dtype=np.int64),
        'input_values': np.array([1 + 2j, 3 + 4j], dtype=np.complex64),
        'input_shape': np.array([2, 2], dtype=np.int64),
        'reduction_axes': np.array([1], dtype=np.int32)
    })
    
    # Input 10
    list_of_inputs.append({
        'keep_dims': False,
        'name': "test10",
        'input_indices': np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64),
        'input_values': np.array([100, 200, 300], dtype=np.int64),
        'input_shape': np.array([2, 2], dtype=np.int64),
        'reduction_axes': np.array([0], dtype=np.int32)
    })
    
    return list_of_inputs

generated_inputs["tf.raw_ops.SparseReduceSum"] = tf_raw_ops_SparseReduceSum_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseSlice_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'name': 'slice1',
        'indices': np.array([[0, 1], [0, 3], [0, 4], [1, 0], [1, 1]], dtype=np.int64),
        'values': np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32),
        'shape': np.array([2, 7], dtype=np.int64),
        'start': np.array([0, 0], dtype=np.int64),
        'size': np.array([2, 4], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'name': 'slice2',
        'indices': np.array([[0, 1], [0, 3], [0, 4], [1, 0], [1, 1]], dtype=np.int64),
        'values': np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32),
        'shape': np.array([2, 7], dtype=np.int64),
        'start': np.array([0, 4], dtype=np.int64),
        'size': np.array([2, 3], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'name': 'slice3',
        'indices': np.array([[1], [3], [5], [8]], dtype=np.int64),
        'values': np.array([1, 2, 3, 4], dtype=np.int32),
        'shape': np.array([10], dtype=np.int64),
        'start': np.array([2], dtype=np.int64),
        'size': np.array([5], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'name': 'slice4',
        'indices': np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.int64),
        'values': np.array([1.5, 2.5, 3.5], dtype=np.float64),
        'shape': np.array([3, 3, 3], dtype=np.int64),
        'start': np.array([1, 1, 1], dtype=np.int64),
        'size': np.array([2, 2, 2], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'name': 'slice5',
        'indices': np.empty((0, 2), dtype=np.int64),
        'values': np.empty((0,), dtype=np.float32),
        'shape': np.array([5, 5], dtype=np.int64),
        'start': np.array([1, 1], dtype=np.int64),
        'size': np.array([3, 3], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'name': 'slice6',
        'indices': np.array([[0, 0, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.int64),
        'values': np.array([100, 200, 300], dtype=np.int64),
        'shape': np.array([2, 2, 2, 2], dtype=np.int64),
        'start': np.array([0, 0, 0, 0], dtype=np.int64),
        'size': np.array([1, 2, 1, 2], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'name': 'slice7',
        'indices': np.array([[10], [50], [55], [60], [90]], dtype=np.int64),
        'values': np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        'shape': np.array([100], dtype=np.int64),
        'start': np.array([50], dtype=np.int64),
        'size': np.array([10], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'name': 'slice8',
        'indices': np.array([[2, 2], [3, 3], [4, 4], [5, 5]], dtype=np.int64),
        'values': np.array([True, False, True, False], dtype=np.bool_),
        'shape': np.array([10, 10], dtype=np.int64),
        'start': np.array([2, 2], dtype=np.int64),
        'size': np.array([5, 5], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'name': 'slice9',
        'indices': np.array([[1, 2, 3]], dtype=np.int64),
        'values': np.array([42], dtype=np.int32),
        'shape': np.array([5, 5, 5], dtype=np.int64),
        'start': np.array([0, 0, 0], dtype=np.int64),
        'size': np.array([5, 5, 5], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'name': 'slice10',
        'indices': np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int64),
        'values': np.array([1, 2, 3, 4], dtype=np.int64),
        'shape': np.array([4, 4], dtype=np.int64),
        'start': np.array([1, 0], dtype=np.int64),
        'size': np.array([2, 4], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSlice"] = tf_raw_ops_SparseSlice_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_required_space_to_batch_paddings_inputs():
    list_of_inputs = []
    
    # Input 1
    input_dict = {
        'input_shape': np.array([10], dtype=np.int32),
        'block_shape': np.array([3], dtype=np.int32),
        'base_paddings': np.array([[1, 2]], dtype=np.int32),
        'name': "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input_dict = {
        'input_shape': np.array([10, 20], dtype=np.int32),
        'block_shape': np.array([3, 4], dtype=np.int32),
        'base_paddings': np.array([[0, 0], [1, 1]], dtype=np.int32),
        'name': "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input_dict = {
        'input_shape': np.array([14, 15, 16], dtype=np.int32),
        'block_shape': np.array([2, 3, 4], dtype=np.int32),
        'base_paddings': np.array([[0, 1], [2, 0], [1, 1]], dtype=np.int32),
        'name': "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_dict = {
        'input_shape': np.array([5], dtype=np.int32),
        'block_shape': np.array([5], dtype=np.int32),
        'base_paddings': np.array([[0, 0]], dtype=np.int32),
        'name': "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_dict = {
        'input_shape': np.array([100, 200, 300, 400], dtype=np.int32),
        'block_shape': np.array([7, 8, 9, 10], dtype=np.int32),
        'base_paddings': np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32),
        'name': "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_dict = {
        'input_shape': np.array([1, 1], dtype=np.int32),
        'block_shape': np.array([10, 10], dtype=np.int32),
        'base_paddings': np.array([[5, 5], [2, 3]], dtype=np.int32),
        'name': "test6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_dict = {
        'input_shape': np.array([1000], dtype=np.int32),
        'block_shape': np.array([1], dtype=np.int32),
        'base_paddings': np.array([[0, 0]], dtype=np.int32),
        'name': "test7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_dict = {
        'input_shape': np.array([7, 11, 13], dtype=np.int32),
        'block_shape': np.array([5, 5, 5], dtype=np.int32),
        'base_paddings': np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32),
        'name': "test8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_dict = {
        'input_shape': np.array([100, 100], dtype=np.int32),
        'block_shape': np.array([99, 99], dtype=np.int32),
        'base_paddings': np.array([[10, 10], [20, 20]], dtype=np.int32),
        'name': "test9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_dict = {
        'input_shape': np.array([2, 4, 6, 8, 10], dtype=np.int32),
        'block_shape': np.array([3, 3, 3, 3, 3], dtype=np.int32),
        'base_paddings': np.array([[1, 1], [2, 2], [0, 0], [1, 2], [2, 1]], dtype=np.int32),
        'name': "test10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.required_space_to_batch_paddings"] = tf_required_space_to_batch_paddings_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_slice_inputs():
    list_of_inputs = []

    # 1. 1D float32 array
    input_dict = {
        "input_": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "begin": np.array([1], dtype=np.int32),
        "size": np.array([3], dtype=np.int32),
        "name": "slice_1d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2. 2D int32 array, with -1 in size
    input_dict = {
        "input_": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        "begin": np.array([0, 1], dtype=np.int32),
        "size": np.array([2, -1], dtype=np.int32),
        "name": "slice_2d_neg_size"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3. 3D int32 array
    input_dict = {
        "input_": np.array([[[1, 1, 1], [2, 2, 2]],
                            [[3, 3, 3], [4, 4, 4]],
                            [[5, 5, 5], [6, 6, 6]]], dtype=np.int32),
        "begin": np.array([1, 0, 0], dtype=np.int64),
        "size": np.array([1, 2, 3], dtype=np.int64),
        "name": "slice_3d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4. 2D float64 array, full slice with -1
    input_dict = {
        "input_": np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64),
        "begin": np.array([0, 0], dtype=np.int32),
        "size": np.array([-1, -1], dtype=np.int32),
        "name": "slice_2d_full"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5. 4D float32 array
    input_dict = {
        "input_": np.arange(16, dtype=np.float32).reshape((2, 2, 2, 2)),
        "begin": np.array([0, 1, 0, 1], dtype=np.int32),
        "size": np.array([1, 1, 2, 1], dtype=np.int32),
        "name": "slice_4d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6. 1D float32 array, size -1
    input_dict = {
        "input_": np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32),
        "begin": np.array([2], dtype=np.int32),
        "size": np.array([-1], dtype=np.int32),
        "name": "slice_1d_neg_size"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7. 3D int32 array, small dimensions
    input_dict = {
        "input_": np.array([[[1], [2]], [[3], [4]]], dtype=np.int32),
        "begin": np.array([0, 0, 0], dtype=np.int32),
        "size": np.array([2, 1, 1], dtype=np.int32),
        "name": "slice_3d_small"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8. 2D boolean array
    input_dict = {
        "input_": np.array([[True, False], [False, True]], dtype=np.bool_),
        "begin": np.array([1, 0], dtype=np.int32),
        "size": np.array([1, 2], dtype=np.int32),
        "name": "slice_2d_bool"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9. 3D uint8 array (image-like)
    input_dict = {
        "input_": np.ones((10, 10, 3), dtype=np.uint8),
        "begin": np.array([2, 2, 0], dtype=np.int32),
        "size": np.array([5, 5, 3], dtype=np.int32),
        "name": "slice_image"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10. 2D float32 array
    np.random.seed(42)
    input_dict = {
        "input_": np.random.randn(5, 5).astype(np.float32),
        "begin": np.array([1, 1], dtype=np.int32),
        "size": np.array([3, 2], dtype=np.int32),
        "name": "slice_random_2d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11. 1D int64 array
    input_dict = {
        "input_": np.array([10, 20, 30, 40], dtype=np.int64),
        "begin": np.array([3], dtype=np.int64),
        "size": np.array([1], dtype=np.int64),
        "name": "slice_1d_int64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.slice"] = tf_slice_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_squeeze_inputs():
    list_of_inputs = []

    # Input 1
    input_val = np.random.randn(1, 2, 1).astype(np.float32)
    axis = [0]
    name = "squeeze_1"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis,
        "name": name
    })

    # Input 2
    input_val = np.random.randint(0, 10, size=(1, 1, 3, 1)).astype(np.int32)
    axis = [0, 1]
    name = "squeeze_2"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis,
        "name": name
    })

    # Input 3
    input_val = np.random.choice([True, False], size=(1, 5)).astype(np.bool_)
    axis = [0]
    name = "squeeze_3"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis,
        "name": name
    })

    # Input 4
    input_val = np.random.randn(2, 1, 3, 1, 4).astype(np.float64)
    axis = [1, 3]
    name = "squeeze_4"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis,
        "name": name
    })

    # Input 5
    input_val = np.random.randint(-5, 5, size=(1,)).astype(np.int64)
    axis = [0]
    name = "squeeze_5"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis,
        "name": name
    })

    # Input 6
    input_val = np.random.randn(1, 2, 1).astype(np.float32)
    axis = [-1]
    name = "squeeze_6"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis,
        "name": name
    })

    # Input 7
    input_val = np.random.randn(1, 1, 1, 1, 1, 1).astype(np.float32)
    axis = [0, 1, 2, 3, 4, 5]
    name = "squeeze_7"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis,
        "name": name
    })

    # Input 8
    real = np.random.randn(1, 4, 1, 4).astype(np.float32)
    imag = np.random.randn(1, 4, 1, 4).astype(np.float32)
    input_val = (real + 1j * imag).astype(np.complex64)
    axis = [0, 2]
    name = "squeeze_8"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis,
        "name": name
    })

    # Input 9
    input_val = np.random.randint(0, 255, size=(3, 1, 2)).astype(np.uint8)
    axis = [1]
    name = "squeeze_9"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis,
        "name": name
    })

    # Input 10
    input_val = np.random.randn(1, 2, 1, 1, 3).astype(np.float32)
    axis = [-3, -2]
    name = "squeeze_10"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis,
        "name": name
    })

    return list_of_inputs

generated_inputs["tf.squeeze"] = tf_squeeze_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_strings_as_string_inputs():
    list_of_inputs = []

    # Input 1
    list_of_inputs.append({
        "input": np.array([1, 2, 3], dtype=np.int32),
        "precision": -1,
        "scientific": False,
        "shortest": False,
        "width": -1,
        "fill": "",
        "name": "as_string_1"
    })

    # Input 2
    list_of_inputs.append({
        "input": np.array([3.14159, 2.71828], dtype=np.float32),
        "precision": 2,
        "scientific": False,
        "shortest": False,
        "width": -1,
        "fill": "",
        "name": "as_string_2"
    })

    # Input 3
    list_of_inputs.append({
        "input": np.array([-10, 0, 100], dtype=np.int64),
        "precision": -1,
        "scientific": False,
        "shortest": False,
        "width": 4,
        "fill": "0",
        "name": "as_string_3"
    })

    # Input 4
    list_of_inputs.append({
        "input": np.array([123.456, 789.012], dtype=np.float64),
        "precision": 4,
        "scientific": True,
        "shortest": False,
        "width": 10,
        "fill": " ",
        "name": "as_string_4"
    })

    # Input 5
    list_of_inputs.append({
        "input": np.array([True, False, True], dtype=np.bool_),
        "precision": -1,
        "scientific": False,
        "shortest": False,
        "width": -1,
        "fill": "",
        "name": "as_string_5"
    })

    # Input 6
    list_of_inputs.append({
        "input": np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32),
        "precision": 1,
        "scientific": False,
        "shortest": True,
        "width": -1,
        "fill": "",
        "name": "as_string_6"
    })

    # Input 7
    list_of_inputs.append({
        "input": np.array([10000, 20000], dtype=np.int32),
        "precision": -1,
        "scientific": False,
        "shortest": False,
        "width": 6,
        "fill": "0",
        "name": "as_string_7"
    })

    # Input 8
    list_of_inputs.append({
        "input": np.array([1e-10, 1e10], dtype=np.float64),
        "precision": 3,
        "scientific": True,
        "shortest": False,
        "width": 12,
        "fill": " ",
        "name": "as_string_8"
    })

    # Input 9
    list_of_inputs.append({
        "input": np.array([-5, 5], dtype=np.int32),
        "precision": -1,
        "scientific": False,
        "shortest": False,
        "width": 3,
        "fill": "0",
        "name": "as_string_9"
    })

    # Input 10
    list_of_inputs.append({
        "input": np.array([1.1, 2.2, 3.3], dtype=np.float32),
        "precision": 0,
        "scientific": False,
        "shortest": False,
        "width": -1,
        "fill": "",
        "name": "as_string_10"
    })

    return list_of_inputs

generated_inputs["tf.strings.as_string"] = tf_strings_as_string_inputs()

import tensorflow as tf
import numpy as np
import copy

class custom_list(list):
    @property
    def shape(self):
        return (len(self),)
    @property
    def dtype(self):
        return np.int32
    @property
    def ndim(self):
        return 1

def tf_strings_format_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'template': "tensor a: {}",
        'inputs': custom_list([np.array([1, 2, 3], dtype=np.int32)]),
        'placeholder': "{}",
        'summarize': 3,
        'name': "format_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'template': "a: {}, b: {}",
        'inputs': custom_list([np.array([1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int32)]),
        'placeholder': "{}",
        'summarize': 2,
        'name': "format_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'template': "val1: %s, val2: %s",
        'inputs': custom_list([np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32), np.array([-1, -2], dtype=np.int32)]),
        'placeholder': "%s",
        'summarize': -1,
        'name': "format_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'template': "Matrix is <X>",
        'inputs': custom_list([np.ones((3, 3, 3), dtype=np.float32)]),
        'placeholder': "<X>",
        'summarize': 1,
        'name': "format_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'template': "empty: {}",
        'inputs': custom_list([np.array([], dtype=np.int32)]),
        'placeholder': "{}",
        'summarize': 3,
        'name': "format_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'template': "bools: [BOOL]",
        'inputs': custom_list([np.array([True, False, True], dtype=bool)]),
        'placeholder': "[BOOL]",
        'summarize': 5,
        'name': "format_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'template': "string tensor: {}",
        'inputs': custom_list([np.array([b"hello", b"world"], dtype=object)]),
        'placeholder': "{}",
        'summarize': 2,
        'name': "format_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'template': "a: {} b: {} c: {}",
        'inputs': custom_list([np.array([1], dtype=np.int32), np.array([2], dtype=np.int32), np.array([3], dtype=np.int32)]),
        'placeholder': "{}",
        'summarize': 3,
        'name': "format_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'template': "high_dim: {}",
        'inputs': custom_list([np.zeros((2, 2, 2, 2), dtype=np.int64)]),
        'placeholder': "{}",
        'summarize': -1,
        'name': "format_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'template': "int32 and float64: {} and {}",
        'inputs': custom_list([np.array([1, 2, 3], dtype=np.int32), np.array([1.1, 2.2], dtype=np.float64)]),
        'placeholder': "{}",
        'summarize': 10,
        'name': "format_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.strings.format"] = tf_strings_format_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_unravel_index_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D indices, 2D dims, int32
    indices1 = np.array([2, 5, 7], dtype=np.int32)
    dims1 = np.array([3, 3], dtype=np.int32)
    list_of_inputs.append({
        "indices": indices1,
        "dims": dims1,
        "name": "unravel_1"
    })

    # Input 2: 0-D index, 2D dims, int32
    indices2 = np.array(4, dtype=np.int32)
    dims2 = np.array([3, 3], dtype=np.int32)
    list_of_inputs.append({
        "indices": indices2,
        "dims": dims2,
        "name": "unravel_2"
    })

    # Input 3: 1D indices, 3D dims, int64
    indices3 = np.array([0, 7, 11], dtype=np.int64)
    dims3 = np.array([2, 2, 3], dtype=np.int64)
    list_of_inputs.append({
        "indices": indices3,
        "dims": dims3,
        "name": "unravel_3"
    })

    # Input 4: 0-D index, 3D dims, int64
    indices4 = np.array(5, dtype=np.int64)
    dims4 = np.array([2, 3, 2], dtype=np.int64)
    list_of_inputs.append({
        "indices": indices4,
        "dims": dims4,
        "name": "unravel_4"
    })

    # Input 5: 1D indices, 1D dims, int32
    indices5 = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    dims5 = np.array([5], dtype=np.int32)
    list_of_inputs.append({
        "indices": indices5,
        "dims": dims5,
        "name": "unravel_5"
    })

    # Input 6: 1D indices, 4D dims, int32
    indices6 = np.array([10, 20, 30], dtype=np.int32)
    dims6 = np.array([2, 3, 4, 2], dtype=np.int32)
    list_of_inputs.append({
        "indices": indices6,
        "dims": dims6,
        "name": "unravel_6"
    })

    # Input 7: 1D indices, large values, int64
    indices7 = np.array([1000000, 2000000], dtype=np.int64)
    dims7 = np.array([3000, 3000], dtype=np.int64)
    list_of_inputs.append({
        "indices": indices7,
        "dims": dims7,
        "name": "unravel_7"
    })

    # Input 8: 0-D index, large dims, int64
    indices8 = np.array(8000000, dtype=np.int64)
    dims8 = np.array([2000, 2000, 5], dtype=np.int64)
    list_of_inputs.append({
        "indices": indices8,
        "dims": dims8,
        "name": "unravel_8"
    })

    # Input 9: 1D indices, 2D dims, int32
    indices9 = np.arange(12, dtype=np.int32)
    dims9 = np.array([3, 4], dtype=np.int32)
    list_of_inputs.append({
        "indices": indices9,
        "dims": dims9,
        "name": "unravel_9"
    })

    # Input 10: 1D indices, 5D dims, int64
    indices10 = np.array([1, 15, 31], dtype=np.int64)
    dims10 = np.array([2, 2, 2, 2, 2], dtype=np.int64)
    list_of_inputs.append({
        "indices": indices10,
        "dims": dims10,
        "name": "unravel_10"
    })

    return list_of_inputs

generated_inputs["tf.unravel_index"] = tf_unravel_index_inputs()

