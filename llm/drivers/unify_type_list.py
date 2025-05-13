import numpy as np
import torch
import tensorflow as tf

def torch_version(input_dict, cpu=True):

    input_list = [torch.tensor(x) for x in input_dict["input"]]
    dtypes = [x.dtype for x in input_list]

    if not cpu:
        input_list = [x.cuda() for x in input_list]

    if all(d == dtypes[0] for d in dtypes):
      result = dtypes[0]
    else:
      max_size = 0
      result = None
      for d in dtypes:
          if d.itemsize > max_size:
              max_size = d.itemsize
              result = d
    
    return {"result": str(result).split('.')[-1].replace(")", "")}

def tensorflow_version(input_dict, cpu=True):

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_list = [tf.constant(x) for x in input_dict["input"]]
        dtypes = [x.dtype for x in input_list]
        
        if all(d == dtypes[0] for d in dtypes):
            result = dtypes[0]
        else:
            max_size = 0
            result = None
            for d in dtypes:
                if tf.as_dtype(d).size > max_size:
                    max_size = tf.as_dtype(d).size
                    result = d
    
    return {"result": str(result).split(" ")[-1].replace(">", "")}

def main():
    A_TOL = 0.01
    input_data = {
        "input": [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([4, 5, 6], dtype=np.float32)
        ]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == tf_result["result"], "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()