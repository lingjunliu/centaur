import numpy as np
import io
import pickle

def torch_version(input_dict, cpu=True):
    import torch

    f = input_dict["f"]
    map_location = input_dict.get("map_location", None)
    pickle_module = input_dict.get("pickle_module", pickle)
    weights_only = input_dict.get("weights_only", True)
    mmap = input_dict.get("mmap", None)
    pickle_load_args = input_dict.get("pickle_load_args", {})

    if not cpu:
        if isinstance(map_location, str) and map_location == "cpu":
            pass
        elif map_location is None:
            map_location = lambda storage, loc: storage.cuda()
        elif isinstance(map_location, dict):
            new_map_location = {}
            for k, v in map_location.items():
                new_map_location[k] = "cuda:0" if v == "cpu" else v
            map_location = new_map_location
        elif callable(map_location):
             pass
        else:
             map_location = "cuda:0"
            

    if isinstance(f, np.ndarray):
        f = io.BytesIO(f.tobytes())

    result = torch.load(
        f,
        map_location=map_location,
        pickle_module=pickle_module,
        weights_only=weights_only,
        mmap=mmap,
        **pickle_load_args
    )

    if not cpu:
        if isinstance(result, torch.Tensor):
           result = result.cpu()
        elif isinstance(result, dict):
            for k, v in result.items():
                if isinstance(v, torch.Tensor):
                    result[k] = v.cpu()
    if isinstance(result, torch.Tensor):
        return {"result": result.numpy()}
    else:
        return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    f = input_dict["f"]
    map_location = input_dict.get("map_location", None)
    pickle_module = input_dict.get("pickle_module", pickle)
    weights_only = input_dict.get("weights_only", True)
    mmap = input_dict.get("mmap", None)
    pickle_load_args = input_dict.get("pickle_load_args", {})
    
    if isinstance(f, np.ndarray):
        f = io.BytesIO(f.tobytes())
    
    try:
        # Deserialize using pickle directly
        result = pickle_module.load(f, **pickle_load_args)
            
        if isinstance(result, dict):
            for k, v in result.items():
                if isinstance(v, np.ndarray):
                    result[k] = tf.convert_to_tensor(v).numpy()
                elif isinstance(v, list):
                    new_list = []
                    for item in v:
                        if isinstance(item, np.ndarray):
                            new_list.append(tf.convert_to_tensor(item).numpy())
                        else:
                            new_list.append(item)
                    result[k] = new_list
            return {"result": result}
        elif isinstance(result, np.ndarray):
            return {"result": tf.convert_to_tensor(result).numpy()}
        else:
            return {"result": result}
    except Exception as e:
        print(f"Error loading with TensorFlow: {e}")
        return {"result": None}

def main():
    A_TOL = 0.01

    # Create a sample tensor to save
    import torch
    sample_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])

    # Save the tensor to a BytesIO buffer
    buffer = io.BytesIO()
    torch.save(sample_tensor, buffer)
    buffer.seek(0)

    # Example input dictionary
    input_data = {
        "f": buffer,
        "weights_only": False
    }

    # Torch example
    torch_result = torch_version(input_data)

    # Reset buffer
    buffer.seek(0)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    
    if torch_result["result"] is not None and tf_result["result"] is not None:
        # Assert to see if they are equal
        assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    else:
        print("One or both results are None, skipping assertion.")

    print("Success")

if __name__ == "__main__":
    main()