import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    try:
        if not torch.backends.vulkan.is_available():
            return {"result": False}
    except AttributeError:
        return {"result": False}
    
    if not cpu:
        torch.cuda.init()
    
    return {"result": torch.backends.vulkan.is_available()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    
    try:
        if not cpu:
            devices = tf.config.list_physical_devices('GPU')
            if devices:
                try:
                    for device in devices:
                        tf.config.experimental.set_memory_growth(device, True)
                    _ = tf.constant([1.0, 2.0, 3.0])
                    return {"result": True}
                except:
                    return {"result": False}
            else:
                return {"result": False}
        else:
            return {"result": False}

    except Exception as e:
        return {"result": False}

def main():
    A_TOL = 0.01

    input_data = {}

    torch_result = torch_version(input_data)

    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == tf_result["result"], "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()