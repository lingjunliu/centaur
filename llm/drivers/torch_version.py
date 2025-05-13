import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.__version__
    
    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    result = tf.__version__
    
    return {"result": np.array(result)}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    torch_version_str = torch_result["result"].item()
    tf_version_str = tf_result["result"].item()
    
    assert isinstance(torch_version_str, str)
    assert isinstance(tf_version_str, str)

    print("Success")

if __name__ == "__main__":
    main()