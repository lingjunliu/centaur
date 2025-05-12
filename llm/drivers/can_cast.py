import numpy as np
import torch
import tensorflow as tf


def torch_version(input_dict, cpu=True):
    from_dtype = input_dict["from"]
    to_dtype = input_dict["to"]

    if not cpu:
        torch.set_default_device("cuda")

    result = torch.can_cast(from_dtype, to_dtype)

    if not cpu:
        torch.set_default_device("cpu")

    return {"result": np.array(result)}


def tensorflow_version(input_dict, cpu=True):
    from_dtype_str = input_dict["from"]
    to_dtype_str = input_dict["to"]

    dtype_mapping = {
        "float64": tf.float64,
        "float32": tf.float32,
        "float16": tf.float16,
        "int64": tf.int64,
        "int32": tf.int32,
        "int16": tf.int16,
        "int8": tf.int8,
        "uint8": tf.uint8,
        "bool": tf.bool,
        "bfloat16": tf.bfloat16
    }

    from_tf_dtype = dtype_mapping.get(from_dtype_str)
    to_tf_dtype = dtype_mapping.get(to_dtype_str)
    
    if from_tf_dtype is None or to_tf_dtype is None:
        return {"result": np.array(False)}


    def tf_can_cast(from_tf_dtype, to_tf_dtype):
        try:
            casted = tf.cast(tf.constant([1], dtype=from_tf_dtype), to_tf_dtype)
            return True
        except tf.errors.InvalidArgumentError:
            return False
        except tf.experimental.numpy.core.ops.InvalidDTypeError:
            return False
        except ValueError:
            return False

    result = tf_can_cast(from_tf_dtype, to_tf_dtype)

    return {"result": np.array(result)}


def main():
    A_TOL = 0.01

    test_cases = [
        {"from": torch.int32, "to": torch.float32},
        {"from": torch.float64, "to": torch.int8},
        {"from": torch.uint8, "to": torch.float64},
        {"from": torch.bool, "to": torch.int32},
        {"from": torch.int32, "to": torch.bool},
        {"from": torch.float32, "to": torch.float64},
        {"from": torch.int8, "to": torch.int32},
        {"from": torch.bfloat16, "to": torch.float32},
        {"from": torch.float32, "to": torch.bfloat16}
    ]

    tf_dtype_mapping = {
        torch.float64: "float64",
        torch.float32: "float32",
        torch.float16: "float16",
        torch.int64: "int64",
        torch.int32: "int32",
        torch.int16: "int16",
        torch.int8: "int8",
        torch.uint8: "uint8",
        torch.bool: "bool",
        torch.bfloat16: "bfloat16"
    }
    
    for input_data in test_cases:
      torch_result = torch_version(input_data)

      tf_input_data = {}
      tf_input_data["from"] = tf_dtype_mapping.get(input_data["from"])
      tf_input_data["to"] = tf_dtype_mapping.get(input_data["to"])
      
      if tf_input_data["from"] is None or tf_input_data["to"] is None:
        continue

      tf_result = tensorflow_version(tf_input_data)


      assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), f"Results do not match for {input_data}"

    print("Success")


if __name__ == "__main__":
    main()