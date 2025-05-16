import numpy as np
from typing import Dict, Optional

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from typing import Dict

    the_type = input_dict["the_type"]
    the_value = input_dict["the_value"]

    if isinstance(the_value, np.ndarray):
        the_value = torch.tensor(the_value)

    if not cpu:
        if isinstance(the_value, torch.Tensor):
            the_value = the_value.cuda()

    result = torch.jit.annotate(the_type, the_value)

    if not cpu:
        if isinstance(result, torch.Tensor):
            result = result.cpu()
    
    if isinstance(result, torch.Tensor):
        return {"result": result.numpy()}
    else:
        return {"result": result}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    from typing import Dict
    import numpy as np

    the_type = input_dict["the_type"]
    the_value = input_dict["the_value"]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        if isinstance(the_value, np.ndarray):
            the_value = tf.constant(the_value)

        result = the_value  # tf has no direct equivalent, pass value as is


        if isinstance(result, tf.Tensor):
            result = result.numpy()
        
        return {"result": result}

def main():
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    A_TOL = 0.01

    input_data_empty_dict = {
        "the_type": Dict[str, int],
        "the_value": {}
    }
    
    input_data_optional = {
        "the_type": Optional[int],
        "the_value": 10
    }

    # Torch example
    torch_result_empty_dict = torch_version(input_data_empty_dict)
    torch_result_optional = torch_version(input_data_optional)
    
    # TensorFlow example
    tf_result_empty_dict = tensorflow_version(input_data_empty_dict)
    tf_result_optional = tensorflow_version(input_data_optional)

    assert torch_result_empty_dict["result"] == tf_result_empty_dict["result"], "Results do not match for empty dict"
    assert torch_result_optional["result"] == tf_result_optional["result"], "Results do not match for Optional type"
    
    input_data_tensor = {
        "the_type": torch.Tensor,
        "the_value": np.array([1, 2, 3], dtype=np.float32)
    }
    
    torch_result_tensor = torch_version(input_data_tensor)
    tf_result_tensor = tensorflow_version(input_data_tensor)
    assert np.allclose(torch_result_tensor["result"], tf_result_tensor["result"], atol=A_TOL), "Results do not match for Tensor"
    

    print("Success")

if __name__ == "__main__":
    main()