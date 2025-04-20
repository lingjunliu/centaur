import torch
import tensorflow as tf
import numpy as np

from src.setseed import set_seed
from src.type_mapping_torch import np_to_torch

# Custom function to promote types in TensorFlow
def tf_promote_types(type1, type2):
    tf_types = {
        tf.float16: 0,
        tf.float32: 1,
        tf.float64: 2,
        tf.int8: 3,
        tf.uint8: 4,
        tf.int16: 5,
        tf.uint16: 6,
        tf.int32: 7,
        tf.uint32: 8,
        tf.int64: 9,
        tf.bool: 10
    }
    inv_tf_types = {v: k for k, v in tf_types.items()}
    
    type1_rank = tf_types[type1]
    type2_rank = tf_types[type2]
    
    return inv_tf_types[max(type1_rank, type2_rank)]

# Function to convert TensorFlow dtype to string equivalent in Torch format
def tf_dtype_to_torch_dtype_string(dtype):
    mapping = {
        tf.float16: "torch.float16",
        tf.float32: "torch.float32",
        tf.float64: "torch.float64",
        tf.int8: "torch.int8",
        tf.uint8: "torch.uint8",
        tf.int16: "torch.int16",
        tf.uint16: "torch.uint16",
        tf.int32: "torch.int32",
        tf.uint32: "torch.uint32",
        tf.int64: "torch.int64",
        tf.bool: "torch.bool"
    }
    return mapping[dtype]

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()
    
    # Unpack input dictionary
    type1 = np_to_torch(input["type1"])
    type2 = np_to_torch(input["type2"])

    # Apply torch.promote_types
    result_type = torch.promote_types(type1, type2)
    
    return {"promote_types_result": str(result_type)}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()
    
    # Unpack input dictionary
    type1 = tf.as_dtype(input["type1"])
    type2 = tf.as_dtype(input["type2"])
    
    # Apply TensorFlow equivalent (custom implementation in this case)
    result_type = tf_promote_types(type1, type2)
    
    return {"promote_types_result": tf_dtype_to_torch_dtype_string(result_type)}

def main():
    # Example input
    input_data = {
        "type1": np.float32,
        "type2": np.float64
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)
    
    # TensorFlow example
    tf_input_data = {
        "type1": np.float32,
        "type2": np.float64
    }
    tf_result = tensorflow_version(tf_input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    if torch_result["promote_types_result"] == tf_result["promote_types_result"]:
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()