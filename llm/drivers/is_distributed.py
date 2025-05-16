import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    if not torch.distributed.is_available():
        return {"result": False}
    
    if not torch.distributed.is_initialized():
        return {"result": False}
    
    return {"result": True}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    try:
        import horovod.tensorflow as hvd
        hvd.init()
        return {"result": True}
    except ImportError:
        return {"result": False}
    except RuntimeError:
        return {"result": False}
    
def main():
    A_TOL = 0.01
    input_data = {}

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    print("torch_result:", torch_result)
    print("tf_result:", tf_result)
    
    print("Success")

if __name__ == "__main__":
    main()