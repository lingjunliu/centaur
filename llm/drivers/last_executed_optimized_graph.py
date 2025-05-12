import numpy as np
import torch
import tensorflow as tf
import io

def torch_version(input_dict, cpu=True):
    def compiled_function(x):
        return x + 1

    input_tensor = torch.jit.script(compiled_function)
    if not cpu:
        pass
    
    input_tensor(torch.randn(1))
    graph = torch.jit.last_executed_optimized_graph()
    
    result = str(graph).strip()
    
    if not cpu:
        pass
    
    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    def build_graph():
        @tf.function
        def f(x):
            return x + 1.0
        return f

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        f = build_graph()
        concrete_function = f.get_concrete_function(tf.TensorSpec(shape=None, dtype=tf.float32))
        graph = concrete_function.graph.as_graph_def()

        result = str(graph).strip()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {}

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == tf_result["result"]

    print("Success")

if __name__ == "__main__":
    main()