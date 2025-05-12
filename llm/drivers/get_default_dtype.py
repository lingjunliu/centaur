import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    if not cpu:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)

    result = torch.get_default_dtype()
    
    if not cpu:
        torch.set_default_tensor_type(torch.FloatTensor)
    
    return {"result": str(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    result = tf.keras.backend.floatx()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {}

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    torch_result_val = torch_result["result"]
    tf_result_val = tf_result["result"]

    torch_mapping = {
      "torch.float32": "float32"
    }

    tf_mapping = {
        "float32": "float32"
    }

    assert torch_mapping[torch_result_val] == tf_mapping[tf_result_val], "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()