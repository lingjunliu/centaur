import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    like_value = torch.tensor(input_dict["like_value"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        like_value = like_value.cuda()
    
    result = torch.cuda.is_available()

    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    if tf.config.list_physical_devices('GPU'):
        result = True
    else:
        result = False

    return {"result": np.array(result)}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1, 2, 3], dtype=np.float32),
        "like_value": np.array([4, 5, 6], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL)

    print("Success")

if __name__ == "__main__":
    main()