import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

def torch_version(input_dict, cpu=True):
    torch.manual_seed(0)
    result = torch.initial_seed()
    return {"result": np.array([result])}

def tensorflow_version(input_dict, cpu=True):
    tf.random.set_seed(0)
    generator = tf.random.Generator.from_seed(0)
    result = generator.uniform(shape=[1], minval=0, maxval=2**63-1, dtype=tf.int64).numpy()
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {}
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()