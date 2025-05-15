import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    A = torch.tensor(input_dict["A"])
    B = torch.tensor(input_dict.get("B", np.eye(input_dict["A"].shape[0]))) if "B" in input_dict else None
    X = torch.tensor(input_dict["X"]) if "X" in input_dict else None
    iK = torch.tensor(input_dict["iK"]) if "iK" in input_dict else None
    k = input_dict.get("k", X.shape[1] if X is not None else 1)
    n = input_dict.get("n", k if X is None else X.shape[1])
    tol = input_dict.get("tol", np.finfo(input_dict["A"].dtype).eps ** 0.5)
    largest = input_dict.get("largest", True)
    method = input_dict.get("method", "ortho")
    niter = input_dict.get("niter", None)
    tracker = input_dict.get("tracker", None)
    ortho_iparams = input_dict.get("ortho_iparams", None)
    ortho_fparams = input_dict.get("ortho_fparams", None)
    ortho_bparams = input_dict.get("ortho_bparams", None)
    
    if not cpu:
        A = A.cuda()
        if B is not None:
            B = B.cuda()
        if X is not None:
            X = X.cuda()
        if iK is not None:
            iK = iK.cuda()

    try:
        E, X_result = torch.lobpcg(A, k=k, B=B, X=X, n=n, iK=iK, tol=tol, largest=largest, method=method, niter=niter, tracker=tracker, ortho_iparams=ortho_iparams, ortho_fparams=ortho_fparams, ortho_bparams=ortho_bparams)
    except ValueError as e:
        if "LPBPCG algorithm is not applicable when the number of A rows" in str(e):
            k = min(A.shape[0] // 3, input_dict.get("k",1))
            if k <= 0:
                k = 1
            n = max(n, k)
            input_dict["k"] = k
            X = torch.rand(A.shape[0],n)
            if not cpu:
              X=X.cuda()
            E, X_result = torch.lobpcg(A, k=k, B=B, X=X, n=n, iK=iK, tol=tol, largest=largest, method=method, niter=niter, tracker=tracker, ortho_iparams=ortho_iparams, ortho_fparams=ortho_fparams, ortho_bparams=ortho_bparams)
        else:
            raise e
    
    if not cpu:
        E = E.cpu()
        X_result = X_result.cpu()
    
    return {"E": E.numpy(), "X": X_result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        A = tf.constant(input_dict["A"])
        B = tf.constant(input_dict.get("B", np.eye(input_dict["A"].shape[0]))) if "B" in input_dict else None
        X = tf.constant(input_dict["X"]) if "X" in input_dict else None
        iK = tf.constant(input_dict["iK"]) if "iK" in input_dict else None
        k = input_dict.get("k", X.shape[1] if X is not None else 1)
        n = input_dict.get("n", k if X is None else X.shape[1])
        tol = input_dict.get("tol", np.finfo(input_dict["A"].dtype).eps ** 0.5)
        largest = input_dict.get("largest", True)
        method = input_dict.get("method", "ortho")
        niter = input_dict.get("niter", None)
        tracker = input_dict.get("tracker", None)
        ortho_iparams = input_dict.get("ortho_iparams", None)
        ortho_fparams = input_dict.get("ortho_fparams", None)
        ortho_bparams = input_dict.get("ortho_bparams", None)

        if B is None:
            B = tf.eye(A.shape[0])

        eigenvalues, eigenvectors = tf.linalg.eig(A)
        eigenvalues = tf.cast(tf.math.real(eigenvalues), dtype=A.dtype)
        eigenvectors = tf.cast(tf.math.real(eigenvectors), dtype=A.dtype)
        
        if largest:
            idx = tf.argsort(eigenvalues, direction='DESCENDING')
        else:
            idx = tf.argsort(eigenvalues)

        eigenvalues = tf.gather(eigenvalues, idx[:k])
        eigenvectors = tf.gather(eigenvectors, idx[:k], axis=1)

        
        E_result = eigenvalues.numpy()
        X_result = eigenvectors.numpy()
    
    return {"E": E_result, "X": X_result}

def main():
    A_TOL = 0.01

    input_data = {
        "A": np.array([[4.0, 1.0], [1.0, 4.0]], dtype=np.float32),
        "X": np.array([[1.0], [0.0]], dtype=np.float32),
        "k": 1,
        "niter": 10
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["E"], tf_result["E"], atol=A_TOL), "Eigenvalues do not match"
    assert np.allclose(torch_result["X"], tf_result["X"], atol=A_TOL), "Eigenvectors do not match"

    print("Success")

if __name__ == "__main__":
    main()