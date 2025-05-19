import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    cu = torch.jit.CompilationUnit()
    cu.define("def forward(self, x):\n  return x\n")
    
    if not cpu:
        input_tensor = input_tensor.cpu()

    return {"result": input_tensor.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    input_tensor = tf.constant(input_dict["input"])

    return {"result": input_tensor.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()