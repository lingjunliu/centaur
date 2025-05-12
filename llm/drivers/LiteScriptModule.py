import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    model_path = input_dict["model_path"]

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    model = torch.jit.load(model_path)
    model.eval()
    if not cpu:
      model = model.cuda()

    with torch.no_grad():
        result = model(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    model_path = input_dict["model_path"]
    
    model = tf.saved_model.load(model_path)

    if not cpu:
        device_string = "/gpu:0"
    else:
        device_string = "/cpu:0"

    with tf.device(device_string):
        infer = model.signatures["serving_default"]
        result = infer(input=input_tensor)['output_0'].numpy()
    
    return {"result": result}

def main():
    import torch
    import tensorflow as tf
    A_TOL = 0.01

    class MyModule(torch.nn.Module):
        def forward(self, x):
            return x + 1.0

    model = MyModule()
    scripted_module = torch.jit.script(model)
    scripted_module.save("traced_model.ptl")
    torch.jit.save(scripted_module, "traced_model.pt")

    tf.saved_model.save(
        obj=tf.Module(),
        export_dir="tf_model",
        signatures={
            'serving_default':
                tf.function(
                    input_signature=[
                        tf.TensorSpec(shape=(None,), dtype=tf.float32, name='input')])(
                    lambda input: {'output_0': input + 1.0})
        }
    )

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "model_path": "traced_model.ptl"
    }

    torch_result = torch_version(input_data)
    input_data["model_path"] = "tf_model"
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

    import os
    os.remove("traced_model.ptl")
    os.remove("traced_model.pt")
    import shutil
    shutil.rmtree("tf_model")

if __name__ == "__main__":
    main()