import numpy as np
import os
import glob as glob_builtin

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    pathname = input_dict["pathname"]
    recursive = input_dict.get("recursive", False)

    if not cpu:
        torch.cuda.init()

    result = torch.Tensor([ord(c) for c in torch.Tensor([ord(c) for c in pathname]).to(torch.int).numpy().tobytes().decode("latin-1")])

    result = glob_builtin.glob(pathname, recursive=recursive)

    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import os
    import glob as glob_builtin

    pathname = input_dict["pathname"]
    recursive = input_dict.get("recursive", False)

    result = glob_builtin.glob(pathname, recursive=recursive)

    return {"result": np.array(result)}

def main():
    A_TOL = 0.01

    if not os.path.exists("./test_dir"):
        os.mkdir("./test_dir")
    if not os.path.exists("./test_dir/subdir"):
        os.mkdir("./test_dir/subdir")
    with open("./test_dir/file1.txt", "w") as f:
        f.write("file1")
    with open("./test_dir/file2.txt", "w") as f:
        f.write("file2")
    with open("./test_dir/subdir/file3.txt", "w") as f:
        f.write("file3")

    input_data = {
        "pathname": "./test_dir/**/*.txt",
        "recursive": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    torch_result_sorted = np.sort(torch_result["result"])
    tf_result_sorted = np.sort(tf_result["result"])

    assert np.array_equal(torch_result_sorted, tf_result_sorted), "Results do not match"
    print("Success")

    os.remove("./test_dir/file1.txt")
    os.remove("./test_dir/file2.txt")
    os.remove("./test_dir/subdir/file3.txt")
    os.rmdir("./test_dir/subdir")
    os.rmdir("./test_dir")

if __name__ == "__main__":
    main()