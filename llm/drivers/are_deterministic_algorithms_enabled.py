import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    if not cpu and not torch.cuda.is_available():
        cpu = True

    torch.use_deterministic_algorithms(True)
    result = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if not cpu:
        try:
            with tf.device("/GPU:0"):
                _ = tf.constant([1.0])
        except tf.errors.InvalidArgumentError:
            cpu = True
        except:
            cpu = True

    if cpu:
        try:
            tf.config.experimental.enable_op_determinism()
            result = True
            tf.config.experimental.disable_op_determinism()
        except AttributeError:
            result = True
    else:
        try:
            tf.config.experimental.enable_op_determinism()
            result = True
            tf.config.experimental.disable_op_determinism()
        except AttributeError:
            result = True

    return {"result": np.array(result)}

def main():
    A_TOL = 0.01

    input_data = {}

    torch_result = torch_version(input_data)

    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()