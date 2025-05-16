import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
from typing import Any, Dict, List

def torch_version(input_dict, cpu=True):
    obj = input_dict["obj"]
    target_type = input_dict["target_type"]

    if isinstance(obj, np.ndarray):
        obj = torch.tensor(obj)

    if not cpu:
        if isinstance(obj, torch.Tensor):
            obj = obj.cuda()
    
    result = torch.jit.isinstance(obj, target_type)
    
    if not cpu and isinstance(result, torch.Tensor):
        result = result.cpu()
    
    return {"result": np.array(result).astype(bool)}

def tensorflow_version(input_dict, cpu=True):
    obj = input_dict["obj"]
    target_type = input_dict["target_type"]
    
    if isinstance(obj, np.ndarray):
        obj = tf.constant(obj)
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        try:
            if target_type == list[tf.Tensor]:
                if isinstance(obj, list):
                    all_tensors = all(isinstance(x, tf.Tensor) for x in obj)
                else:
                    all_tensors = False
                result = all_tensors
            elif target_type == dict[str, str]:
                if isinstance(obj, dict):
                    all_strings = all(isinstance(k, str) and isinstance(v, str) for k, v in obj.items())
                else:
                    all_strings = False
                result = all_strings
            elif target_type == list[str]:
                if isinstance(obj, list):
                    all_strings = all(isinstance(x, str) for x in obj)
                else:
                    all_strings = False
                result = all_strings
            elif target_type == int:
                result = isinstance(obj, int)
            elif target_type == bool:
                result = isinstance(obj, bool)
            else:
                result = False
        except Exception as e:
            result = False
        result = np.array(result).astype(bool)
    return {"result": result}

def main():
    A_TOL = 0.01
    
    input_data_tensor_list = {
        "obj": [np.random.rand(3, 3).astype(np.float32), np.random.rand(4, 3).astype(np.float32)],
        "target_type": list[tf.Tensor]
    }

    torch_result_tensor_list = torch_version({"obj": [torch.rand(3, 3), torch.rand(4, 3)], "target_type": list[torch.Tensor]})
    tf_result_tensor_list = tensorflow_version(input_data_tensor_list)
    
    assert np.allclose(torch_result_tensor_list["result"], tf_result_tensor_list["result"], atol=A_TOL), "Results do not match"

    input_data_dict = {
        "obj": {"key1":"val1","key2":"val2"},
        "target_type": dict[str,str]
    }

    torch_result_dict = torch_version({"obj": {"key1":"val1","key2":"val2"}, "target_type": dict[str,str]})
    tf_result_dict = tensorflow_version(input_data_dict)
    
    assert np.allclose(torch_result_dict["result"], tf_result_dict["result"], atol=A_TOL), "Results do not match"

    input_data_string_list = {
        "obj": ["a", "b"],
        "target_type": list[str]
    }

    torch_result_string_list = torch_version({"obj": ["a", "b"], "target_type": list[str]})
    tf_result_string_list = tensorflow_version(input_data_string_list)
    
    assert np.allclose(torch_result_string_list["result"], tf_result_string_list["result"], atol=A_TOL), "Results do not match"

    input_data_int = {
        "obj": 1,
        "target_type": int
    }

    torch_result_int = torch_version({"obj": 1, "target_type": int})
    tf_result_int = tensorflow_version(input_data_int)
    
    assert np.allclose(torch_result_int["result"], tf_result_int["result"], atol=A_TOL), "Results do not match"

    input_data_bool = {
        "obj": True,
        "target_type": bool
    }

    torch_result_bool = torch_version({"obj": True, "target_type": bool})
    tf_result_bool = tensorflow_version(input_data_bool)
    
    assert np.allclose(torch_result_bool["result"], tf_result_bool["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()