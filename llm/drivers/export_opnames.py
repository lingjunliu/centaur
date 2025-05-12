import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x + 1

    model = Model()
    traced_script_module = torch.jit.trace(model, input_tensor)
    result = torch.jit.export_opnames(traced_script_module)
    
    if not cpu:
        result = [r.cpu() for r in result]
    
    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    class Model(tf.Module):
        def __init__(self):
            super().__init__()

        @tf.function
        def __call__(self, x):
            return x + 1

    model = Model()
    input_tensor = tf.constant(input_dict["input"])
    concrete_function = model.__call__.get_concrete_function(input_tensor)
    op_names = [op.name for op in concrete_function.graph.get_operations()]
    
    return {"result": op_names}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    torch_result_set = set(torch_result["result"])
    tf_result_set = set(tf_result["result"])
    
    # Filter out "Const" ops from TensorFlow, as they can vary
    tf_result_set = {op for op in tf_result_set if "Const" not in op}
    
    assert torch_result_set == tf_result_set, "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()