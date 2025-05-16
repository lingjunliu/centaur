import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    if not cpu:
        torch.cuda.init()
    
    if "dtype" in input_dict:
      torch_dtype = getattr(torch, input_dict["dtype"])
    else:
      torch_dtype = torch.float32

    if not cpu:
        with torch.cuda.device(0):
            torch.set_default_dtype(torch_dtype)
            torch.set_autocast_cpu_dtype(torch_dtype)
            result = torch.get_autocast_cpu_dtype()
            torch.set_default_dtype(torch.float32)
            torch.set_autocast_cpu_dtype(torch.float32)

    else:
        torch.set_default_dtype(torch_dtype)
        torch.set_autocast_cpu_dtype(torch_dtype)
        result = torch.get_autocast_cpu_dtype()
        torch.set_default_dtype(torch.float32)
        torch.set_autocast_cpu_dtype(torch.float32)

    return {"result": str(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    if "dtype" in input_dict:
      torch_dtype = getattr(torch, input_dict["dtype"])

      if torch_dtype == torch.float16:
          tf_dtype = tf.float16
      elif torch_dtype == torch.bfloat16:
          tf_dtype = tf.bfloat16
      elif torch_dtype == torch.float32:
          tf_dtype = tf.float32
      elif torch_dtype == torch.float64:
          tf_dtype = tf.float64
      elif torch_dtype == torch.complex32:
        tf_dtype = tf.complex64 # No tf.complex32
      elif torch_dtype == torch.complex64:
          tf_dtype = tf.complex64
      elif torch_dtype == torch.complex128:
          tf_dtype = tf.complex128
      else:
          tf_dtype = tf.float32 #Default
    else:
        tf_dtype = tf.float32

    if tf_dtype == tf.float16:
      result = "torch.float16"
    elif tf_dtype == tf.bfloat16:
        result = "torch.bfloat16"
    elif tf_dtype == tf.float32:
        result = "torch.float32"
    elif tf_dtype == tf.float64:
        result = "torch.float64"
    elif tf_dtype == tf.complex64:
      result = "torch.complex64"
    elif tf_dtype == tf.complex128:
      result = "torch.complex128"

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
       "dtype": "float32"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == tf_result["result"], "Results do not match"
    
    input_data = {
       "dtype": "float16"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == tf_result["result"], "Results do not match"

    input_data = {
       "dtype": "bfloat16"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == tf_result["result"], "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()