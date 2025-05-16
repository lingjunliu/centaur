import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    min_val = input_dict["min"]
    max_val = input_dict["max"]
    inclusive = input_dict.get("inclusive", True)

    if not cpu:
        input_tensor = input_tensor.cuda()

    shape = input_tensor.shape
    constrained_shape = []
    for dim in shape:
        constrained_dim = max(min_val, min(dim, max_val)) if inclusive else max(min_val+1, min(dim, max_val-1))
        constrained_shape.append(constrained_dim)

    result = np.zeros(constrained_shape)

    if not cpu:
        result = torch.tensor(result).cuda().cpu().numpy()

    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        min_val = input_dict["min"]
        max_val = input_dict["max"]
        inclusive = input_dict.get("inclusive", True)

        input_shape = tf.shape(input_tensor)
        
        if inclusive:
            constrained_shape = tf.clip_by_value(input_shape, clip_value_min=min_val, clip_value_max=max_val)
        else:
            constrained_shape = tf.clip_by_value(input_shape, clip_value_min=min_val + 1, clip_value_max=max_val - 1)
        
        result = constrained_shape.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([5, 10, 15, 20], dtype=np.int32),
        "min": 1,
        "max": 5,
        "inclusive": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(np.array(torch_result["result"]).shape, tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([5, 10, 15, 20], dtype=np.int32),
        "min": 1,
        "max": 5,
        "inclusive": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(np.array(torch_result["result"]).shape, tf_result["result"], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()