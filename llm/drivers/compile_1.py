import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    model = input_dict["model"]
    fullgraph = input_dict.get("fullgraph", False)
    dynamic = input_dict.get("dynamic", None)
    backend = input_dict.get("backend", 'inductor')
    mode = input_dict.get("mode", None)
    options = input_dict.get("options", None)
    disable = input_dict.get("disable", False)
    
    if not cpu:
      pass 
    
    compiled_model = torch.compile(model, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode, options=options, disable=disable)
    
    example_input = torch.tensor(input_dict["example_input"])
    
    if not cpu:
        example_input = example_input.cuda()

    result = compiled_model(example_input)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    model = input_dict["model"]
    fullgraph = input_dict.get("fullgraph", False)
    dynamic = input_dict.get("dynamic", None)
    backend = input_dict.get("backend", 'inductor')
    mode = input_dict.get("mode", None)
    options = input_dict.get("options", None)
    disable = input_dict.get("disable", False)
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    def wrapped_model(x):
      with tf.device(device_string):
        tf_x = tf.convert_to_tensor(x, dtype=tf.float32)
        return model(tf_x).numpy()
    
    example_input = input_dict["example_input"]

    result = wrapped_model(example_input)

    return {"result": result}

def main():
    A_TOL = 0.01

    def simple_model(x):
        return tf.sin(x) + tf.cos(x)

    def torch_simple_model(x):
        return torch.sin(x) + torch.cos(x)
    
    input_data = {
        "model": simple_model,
        "example_input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "fullgraph": True,
        "options": {}
    }

    torch_result = torch_version({"model": torch_simple_model,
        "example_input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "fullgraph": True,
        "options": {}})
    
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()