import numpy as np
import os

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import glob

    pattern = input_dict["pattern"]
    recursive = input_dict.get("recursive", False)
    
    if not cpu:
        pass
    
    result = glob.glob(pattern, recursive=recursive)
    
    if not cpu:
        pass
    
    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import glob
    import os

    pattern = input_dict["pattern"]
    recursive = input_dict.get("recursive", False)
    
    if not cpu:
        pass
    
    result = glob.glob(pattern, recursive=recursive)
    
    if not cpu:
        pass
    
    return {"result": result}

def main():
    A_TOL = 0.01
    
    os.makedirs("test_dir", exist_ok=True)
    with open("test_dir/file1.txt", "w") as f:
        f.write("test")
    with open("test_dir/file2.txt", "w") as f:
        f.write("test")
    os.makedirs("test_dir/subdir", exist_ok=True)
    with open("test_dir/subdir/file3.txt", "w") as f:
        f.write("test")


    input_data = {
        "pattern": "test_dir/*.txt"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert set(torch_result["result"]) == set(tf_result["result"]), "Results do not match"

    input_data = {
        "pattern": "test_dir/**/*.txt",
        "recursive": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert set(torch_result["result"]) == set(tf_result["result"]), "Results do not match"
    
    import shutil
    shutil.rmtree("test_dir")


    print("Success")

if __name__ == "__main__":
    main()