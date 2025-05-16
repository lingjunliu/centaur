import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    if "size" in input_dict:
        mean = input_dict["mean"]
        std = input_dict["std"]
        size = input_dict["size"]
        if not cpu:
            result = torch.normal(mean, std, size=size)
        else:
            result = torch.normal(mean, std, size=size)
    elif "std" in input_dict and isinstance(input_dict["std"], np.ndarray):
        if "mean" in input_dict and isinstance(input_dict["mean"], float):
            mean = input_dict.get("mean", 0.0)
            std = torch.tensor(input_dict["std"])
            if not cpu:
                std = std.cuda()
            result = torch.normal(mean=mean, std=std)
        else:
            mean = torch.tensor(input_dict["mean"])
            std = torch.tensor(input_dict["std"])
            if not cpu:
                mean = mean.cuda()
                std = std.cuda()

            result = torch.normal(mean=mean, std=std)
    else:
        if "std" in input_dict:
            mean = torch.tensor(input_dict["mean"])
            std = input_dict.get("std", 1.0)
            if not cpu:
                mean = mean.cuda()
            result = torch.normal(mean=mean, std=std)
        else:
            mean = input_dict["mean"]
            std = input_dict["std"]
            size = input_dict["size"]
            result = torch.normal(mean, std, size=size)

    if not cpu:
        result = result.cpu()
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        if "size" in input_dict:
            mean = input_dict["mean"]
            std = input_dict["std"]
            size = input_dict["size"]
            result = tf.random.normal(size, mean=mean, stddev=std)
        elif "std" in input_dict and isinstance(input_dict["std"], np.ndarray):
            if "mean" in input_dict and isinstance(input_dict["mean"], float):
                mean = input_dict.get("mean", 0.0)
                std = tf.constant(input_dict["std"])
                result = tf.random.normal(std.shape, mean=mean, stddev=std)
            else:
                mean = tf.constant(input_dict["mean"])
                std = tf.constant(input_dict["std"])
                result = tf.random.normal(mean.shape, mean=mean, stddev=std)
        else:
            if "std" in input_dict:
                mean = tf.constant(input_dict["mean"])
                std = input_dict.get("std", 1.0)
                result = tf.random.normal(mean.shape, mean=mean, stddev=std)
            else:
                mean = input_dict["mean"]
                std = input_dict["std"]
                size = input_dict["size"]
                result = tf.random.normal(size, mean=mean, stddev=std)

        result = result.numpy()
    return {"result": result}

def main():
    A_TOL = 0.1

    input_data1 = {
        "mean": np.arange(1., 6., dtype=np.float32),
        "std": 1.0
    }
    torch_result1 = torch_version(input_data1)
    tf_result1 = tensorflow_version(input_data1)
    assert np.allclose(torch_result1["result"], tf_result1["result"], rtol=1e-03, atol=A_TOL)
    
    input_data2 = {
        "mean": 2,
        "std": 3,
        "size": (1, 4)
    }
    torch_result2 = torch_version(input_data2)
    tf_result2 = tensorflow_version(input_data2)
    assert np.allclose(torch_result2["result"], tf_result2["result"], rtol=1e-03, atol=A_TOL)

    input_data3 = {
        "mean": 0.5,
        "std": np.arange(1., 6., dtype=np.float32)
    }
    torch_result3 = torch_version(input_data3)
    tf_result3 = tensorflow_version(input_data3)
    assert np.allclose(torch_result3["result"], tf_result3["result"], rtol=1e-03, atol=A_TOL)

    input_data4 = {
        "mean": np.arange(1., 11., dtype=np.float32),
        "std": np.arange(1, 0, -0.1, dtype=np.float32)
    }

    torch_result4 = torch_version(input_data4)
    tf_result4 = tensorflow_version(input_data4)
    assert np.allclose(torch_result4["result"], tf_result4["result"], rtol=1e-03, atol=A_TOL)
    
    print("Success")

if __name__ == "__main__":
    main()