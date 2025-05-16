import numpy as np

# Mapping from PyTorch dtypes to TensorFlow dtypes
TF_DTYPE_MAP = {
    torch.float32: tf.float32,
    torch.float64: tf.float64,
    torch.int32: tf.int32,
    torch.int64: tf.int64,
    torch.uint8: tf.uint8,
    torch.bool: tf.bool,
    # Add other necessary mappings
}

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from_dtype = input["from"]
    to_dtype = input["to"]
    
    # Use PyTorch's can_cast function
    result = torch.can_cast(from_dtype, to_dtype)

    return {"can_cast": result}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    from_dtype = input["from"]
    to_dtype = input["to"]
    
    from_tf_dtype = TF_DTYPE_MAP.get(from_dtype, None)
    to_tf_dtype = TF_DTYPE_MAP.get(to_dtype, None)
    
    def can_cast_tf(from_dtype, to_dtype):
        if from_dtype is None or to_dtype is None:
            return False
        try:
            # Safe casting logic in TensorFlow
            safe_cast = {
                tf.float32: [tf.float32, tf.float64],
                tf.float64: [tf.float32, tf.float64],
                tf.int32: [tf.int32, tf.int64, tf.float32, tf.float64],
                tf.int64: [tf.int64, tf.float64],
                tf.uint8: [tf.uint8, tf.float32, tf.float64],
                tf.bool: [tf.bool],
                # Add other necessary rules
            }
            return to_dtype in safe_cast.get(from_dtype, [])
        except KeyError:
            return False

    result = can_cast_tf(from_tf_dtype, to_tf_dtype)
    return {"can_cast": result}

def main():
    # Example input
    input_data = {
        "from": torch.float32,
        "to": torch.int32,
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    if torch_result["can_cast"] == tf_result["can_cast"]:
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()