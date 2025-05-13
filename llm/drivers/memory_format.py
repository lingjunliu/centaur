import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    memory_format = input_dict.get("memory_format", torch.contiguous_format)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = input_tensor.contiguous(memory_format=memory_format)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_np = input_dict["input"]
    
    if input_np.ndim < 2:
      result = input_np
    else:
      input_tensor = tf.constant(input_np)
      memory_format = input_dict.get("memory_format", "contiguous")

      if memory_format == "contiguous" or memory_format == tf.compat.v1.keras.backend.floatx():
        result = input_np
      elif memory_format == "channels_last":
        result = tf.transpose(input_tensor, perm=[0, 2, 3, 1]).numpy() # NCHW to NHWC
      elif memory_format == "channels_first":
        result = input_np

      else:
          result = input_np
    
    return {"result": result}

def main():
    A_TOL = 0.01
    
    input_data1 = {
        "input": np.array([[0.0202, 1.0985], [1.3506, -0.6056]], dtype=np.float32),
    }

    input_data2 = {
        "input": np.array([[[[0.0202, 1.0985], [1.3506, -0.6056]]]], dtype=np.float32),
    }
    
    torch_result1 = torch_version(input_data1)
    tf_result1 = tensorflow_version(input_data1)
    assert np.allclose(torch_result1["result"], tf_result1["result"], atol=A_TOL), "Results do not match"

    torch_result2 = torch_version(input_data2)
    tf_result2 = tensorflow_version(input_data2)
    assert np.allclose(torch_result2["result"], tf_result2["result"], atol=A_TOL), "Results do not match"

    input_data3 = {
        "input": np.array([[[[0.0202, 1.0985], [1.3506, -0.6056]]]], dtype=np.float32),
        "memory_format": "channels_last"
    }
    input_data4 = {
        "input": np.array([[[[0.0202, 1.0985], [1.3506, -0.6056]]]], dtype=np.float32),
        "memory_format": "channels_first"
    }
    
    print("Success")

if __name__ == "__main__":
    main()