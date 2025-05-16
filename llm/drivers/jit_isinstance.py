import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from typing import Any, Dict, List, Tuple, Optional

    obj = torch.tensor(input_dict["obj"])
    target_type = input_dict["target_type"]

    if isinstance(target_type, str):
        if target_type == "bool":
            target_type = bool
        elif target_type == "int":
            target_type = int
        else:
            raise ValueError(f"Unsupported target type: {target_type}")
    elif target_type == List[torch.Tensor]:
        target_type = List[torch.Tensor]
    elif target_type == Dict[str, str]:
        target_type = Dict[str, str]
    elif target_type == Tuple[int, str, int]:
        target_type = Tuple[int, str, int]
    elif target_type == Optional[Tuple[int,str,int]]:
        target_type = Optional[Tuple[int,str,int]]

    if not cpu:
        obj = obj.cuda()

    try:
        result = isinstance(obj, target_type)
    except Exception as e:
        result = False

    if not cpu:
        pass

    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    from typing import Any, Dict, List, Tuple, Optional

    obj = input_dict["obj"]
    target_type = input_dict["target_type"]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        try:
            if isinstance(target_type, str):
                if target_type == "bool":
                    target_type = bool
                elif target_type == "int":
                    target_type = int
                else:
                    raise ValueError(f"Unsupported target type: {target_type}")
            elif target_type == List[tf.Tensor]:
                target_type = List[tf.Tensor]
            elif target_type == Dict[str, str]:
                target_type = Dict[str, str]
            elif target_type == Tuple[int, str, int]:
                target_type = Tuple[int, str, int]
            elif target_type == Optional[Tuple[int,str,int]]:
                target_type = Optional[Tuple[int,str,int]]
            obj = tf.constant(obj)
            
            if isinstance(target_type, type):
                result = isinstance(obj, target_type)
            else:
                result = False

        except Exception as e:
            result = False

        result = np.array(result)

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "obj": np.array([1, 2, 3], dtype=np.int32),
        "target_type": "int"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "obj": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "target_type": "bool"
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "obj": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "target_type": "int"
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()