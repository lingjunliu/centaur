import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    seed = input_dict["seed"]
    
    if not cpu:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)
    
    return {}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    seed = input_dict["seed"]

    tf.random.set_seed(seed)
    np.random.seed(seed)

    return {}

def main():
    A_TOL = 0.01

    input_data = {
        "seed": 42
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    print("Success")

if __name__ == "__main__":
    main()