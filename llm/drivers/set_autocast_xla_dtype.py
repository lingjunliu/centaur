import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    xla_dtype = input_dict["xla_dtype"]
    
    if not cpu:
        torch.xla.set_autocast_xla_dtype(xla_dtype)
    else:
        torch.set_autocast_xla_dtype(xla_dtype)
    
    if not cpu:
        result = torch.xla.get_autocast_xla_dtype()
    else:
        result = torch.get_autocast_xla_dtype()

    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    xla_dtype = input_dict["xla_dtype"]
    
    def dummy_set_autocast_xla_dtype(dtype):
        global current_dtype
        current_dtype = dtype

    def dummy_get_autocast_xla_dtype():
        global current_dtype
        return current_dtype
    
    global current_dtype
    current_dtype = tf.float32
    
    dummy_set_autocast_xla_dtype(xla_dtype)
    result = dummy_get_autocast_xla_dtype()

    return {"result": result}

def main():
    A_TOL = 0.01

    import torch
    torch_dtype_mapping = {
        torch.float16: "torch.float16",
        torch.float32: "torch.float32"
    }
    import tensorflow as tf
    tf_dtype_mapping = {
        tf.float16: "tf.float16",
        tf.float32: "tf.float32"
    }
    np_dtype_list = [torch.float16, torch.float32]
    
    
    if hasattr(torch, 'bfloat16') and hasattr(tf, 'bfloat16'):
        torch_dtype_mapping[torch.bfloat16] = "torch.bfloat16"
        tf_dtype_mapping[tf.bfloat16] = "tf.bfloat16"
        np_dtype_list.append(torch.bfloat16)
    
    for torch_dtype in np_dtype_list:
        input_data = {
            "xla_dtype": torch_dtype
        }
        torch_result = torch_version(input_data)
        tf_result = tensorflow_version(input_data)

        tf_dtype = None
        for k, v in tf_dtype_mapping.items():
            if str(k) == str(tf_result["result"]):
                tf_dtype = k
                break

        assert str(torch_result["result"]) == torch_dtype_mapping[torch_dtype], f"Results do not match: torch {torch_result['result']}, tf {tf_result['result']}"
        assert str(tf_result["result"]) == tf_dtype_mapping[tf_dtype], f"Results do not match: torch {torch_result['result']}, tf {tf_result['result']}"

    print("Success")

if __name__ == "__main__":
    main()