import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.jit.script(torch.nn.Linear(10, 10))

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.jit.optimize_for_inference(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": str(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    from tensorflow.python.eager.def_function import function as tffunction
    import tensorflow.compat.v1 as tfv1
    
    class Model(tf.Module):
      def __init__(self):
        super(Model, self).__init__()
        self.linear = tf.keras.layers.Dense(units=10,
                                             kernel_initializer='ones',
                                             bias_initializer='zeros')
      @tf.function(input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32)])
      def __call__(self, x):
        return self.linear(x)
    
    model = Model()
    # Convert the Keras model to concrete function.
    concrete_func = model.__call__.get_concrete_function(tf.TensorSpec([1, 10], dtype=tf.float32))

    # Convert the concrete function to graph.
    tfv1.disable_eager_execution()
    graph_def = concrete_func.graph.as_graph_def()

    if not cpu:
        pass
    
    result = str(graph_def)
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 10).astype(np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    print("Success")

if __name__ == "__main__":
    main()