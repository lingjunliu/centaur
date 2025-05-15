import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    if "dtype" in input_dict:
        dtype = input_dict["dtype"]
    else:
        dtype = None
    if not cpu and dtype is not None:
        if dtype == np.float16:
            torch_dtype = torch.float16
        elif dtype == np.float32:
            torch_dtype = torch.float32
        elif dtype == np.float64:
            torch_dtype = torch.float64
        elif dtype == np.int8:
            torch_dtype = torch.int8
        elif dtype == np.int16:
            torch_dtype = torch.int16
        elif dtype == np.int32:
            torch_dtype = torch.int32
        elif dtype == np.int64:
            torch_dtype = torch.int64
        elif dtype == np.uint8:
            torch_dtype = torch.uint8
        else:
            raise ValueError("Unsupported dtype")

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            with torch.cuda.amp.autocast():
                autocast_dtype = torch.get_autocast_dtype('cuda')
                if autocast_dtype == torch.float16:
                    return {'result': 'float16'}
                else:
                    return {'result': 'float32'}
        else:
             return {'result': str(torch.float32)}
    else:
        if torch.cuda.is_available():
            return {'result': str(torch.float32)}
        else:
             return {'result': str(torch.float32)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if "dtype" in input_dict:
        dtype = input_dict["dtype"]
    else:
        dtype = None

    if not cpu and dtype is not None:
      if dtype == np.float16:
          tf_dtype = tf.float16
      elif dtype == np.float32:
          tf_dtype = tf.float32
      elif dtype == np.float64:
          tf_dtype = tf.float64
      elif dtype == np.int8:
          tf_dtype = tf.int8
      elif dtype == np.int16:
          tf_dtype = tf.int16
      elif dtype == np.int32:
          tf_dtype = tf.int32
      elif dtype == np.int64:
          tf_dtype = tf.int64
      elif dtype == np.uint8:
          tf_dtype = tf.uint8
      else:
          raise ValueError("Unsupported dtype")

      if tf.config.list_physical_devices('GPU'):
          policy = tf.keras.mixed_precision.Policy("mixed_float16")
          tf.keras.mixed_precision.set_global_policy(policy)
          if policy.compute_dtype == tf.float16:
              return {'result': 'float16'}
          else:
              return {'result': 'float32'}
      else:
          return {'result': 'float32'}
    else:
        return {'result': 'float32'}

def main():
    A_TOL = 0.01
    # Example input
    input_data = {
        "dtype": np.float16
    }

    # Torch example
    torch_result = torch_version(input_data, cpu=False)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data, cpu=False)
    
    # Assert to see if they are equal
    assert torch_result["result"] == tf_result["result"], "Results do not match"

    input_data = {
        "dtype": np.float32
    }

    # Torch example
    torch_result = torch_version(input_data, cpu=False)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data, cpu=False)
    
    # Assert to see if they are equal
    assert torch_result["result"] == tf_result["result"], "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()