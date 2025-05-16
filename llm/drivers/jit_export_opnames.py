import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x + 1

    model = MyModule()
    traced_script_module = torch.jit.trace(model, input_tensor)
    result = torch.jit.export_opnames(traced_script_module)

    if not cpu:
        result = result

    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.convert_to_tensor(input_dict["input"])
    
    
    @tf.function
    def tf_module(x):
      return x + 1
    
    concrete_function = tf_module.get_concrete_function(input_tensor)
    op_names = [op.name for op in concrete_function.graph.get_operations()]

    return {"result": op_names}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    torch_result_set = set(torch_result["result"])
    tf_result_set = set(tf_result["result"])

    assert torch_result_set == tf_result_set, "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()