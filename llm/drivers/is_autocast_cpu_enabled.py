import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    if not hasattr(torch, 'is_autocast_cpu_enabled'):
        return {'result': False}

    if not cpu:
        torch.cuda.init()

    result = torch.is_autocast_cpu_enabled()

    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    
    if not hasattr(tf.keras.mixed_precision, 'get_global_policy'):
        return {'result': False}

    policy = tf.keras.mixed_precision.get_global_policy()
    
    if policy.compute_dtype == 'float32':
        return {'result': np.array(False)}
    else:
        return {'result': np.array(True)}


def main():
    A_TOL = 0.01
    input_data = {}

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()