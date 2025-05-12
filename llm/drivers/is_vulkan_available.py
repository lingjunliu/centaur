import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    try:
        vulkan_available = torch.backends.vulkan.is_available()
    except AttributeError:
        return {"result": np.array(False)}

    if not vulkan_available:
        return {"result": np.array(False)}

    if not cpu:
        if torch.cuda.is_available():
            torch.cuda.init()

    return {"result": np.array(vulkan_available)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    try:
        if not cpu:
            physical_devices = tf.config.list_physical_devices('GPU')
            if len(physical_devices) == 0:
                return {"result": np.array(False)}

        return {"result": np.array(False)}
    except Exception as e:
        return {"result": np.array(False)}

def main():
    A_TOL = 0.01
    input_data = {}

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL)

    print("Success")

if __name__ == "__main__":
    main()