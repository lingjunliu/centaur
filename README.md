Invariant-based test input generator for Deep Learning Library APIs, e.g., Jax, Pytorch, and TensorFlow. The goal is to find bugs on these APIs.

The tool is structured in two parts:

 1. Inference of input invariants
 2. Invariant-based input generator

The code is organized as follow:

```markdown
- 📁 bug_reports       # code snippets containing the bugs reported 
- 📁 drivers           # driver to call an API in various libraries
- 📁 eval              # code related to evaluation (e.g. oracle, coverage)
- 📁 generator         # invariant-based input generator
- 📁 invariants        # mined invariants from APIs
- 📁 learner           # invariant learner
- 📁 llm               # code to generate drivers and signatures using gemini
- 📄 pytest.ini        # pytest configuration file
- 📄 requirements.txt  # dependencies of this project
- 📄 apis.txt          # list of supported APIs
- 📄 signatures.json   # signatures for the supported APIs
- 📁 scripts           # scripts (e.g., run test, demo generator, etc.)
- 📁 tests             # tests
- 📁 utils             # utility functions
```

<h1>Prerequisites</h1>
 
 - **python**: The tool uses `python 3.12`
 - **venv**: `sudo apt install python3.12-venv`

<h1>Use cases</h1>

<h2> 1. Learn invariants (offline) </h2>

 The `infer_invariants` function in the file `learner/invariant_inference.py` can generate a list of inputs randomly, check which of them are valid and for each valid input, check which rules are satisfied by them. It returns a set of tuples `(arity, rule_name, arg1, arg2, ...)` where `arg1`, `arg2`, ... are the arguments in the input that are relevant for a rule and `arity` is the number of arguments this rule accepts.

 To run invariant inference for a single api, run the following (under the venv):
 ```bash
 (venv) ~/dll-fuzzing-with-input-invariants$ python -m learner.invariant_inference <api> <time budget> <1 to regenerate invariants 0 otherwise>
 ```
 Example:
 ```bash
 (venv) ~/dll-fuzzing-with-input-invariants$ python -m learner.invariant_inference combinations 300 1
 ```
 This will generate (regenerate if already exists since `1` is passed as `regen`) the invariants for the api `combinations` and it will use a time budget of `300` seconds to do so.

 The tests written under `tests/test_invariants.py` demonstrates usage of this function.

<h2> 2. Generate models (offline) </h2>

 To generate models by solving the constraints, the script `scripts/generate_models_with_slurm.sh` needs to be used. **Be sure to install and configure slurm before running this.**. This runs model generation for all variations of the apis from `torch_variations.txt` for PyTorch and `tf_variations.txt` for Tensorflow. Since this is an offline mode, running this once is enough to run online fuzzing campaigns.

 ```bash
 (venv) ~/dll-fuzzing-with-input-invariants$ bash scripts/generate_models_with_slurm.sh <duration> <n_max> <lib> <seed> <regen>
 ```
 - `duration`: Time budget for generating model for each api variation in seconds.
 - `n_max`: Passing 0 (default) means no max on number of models. Anything `> 0` will limit the number of models to that number (if it can reach that number before the time budget `duration` runs out).
 - `lib`: `torch` for PyTorch, `tf` for Tenosrflow
 - `seed`: Seed for the generator, default `200`.
 - `regen`: Pass 1 to regenerate models that already exist. Default: 0.

 Example:
 ```bash
 (venv) ~/dll-fuzzing-with-input-invariants$ bash scripts/generate_models_with_slurm.sh 3600 1000 torch 42 1
 ```
 This will generate models for each variation of torch apis until 1h passes or 1000 max models are generated, even if models exist. `42` will be used as the seed.

<h2> 3. Generate inputs (online) </h2>

 The `scripts/run_harness.sh` can demonstrate running input generation for some example apis. To run this:
 ```bash
 (venv) ~/dll-fuzzing-with-input-invariants$ bash run_harness.sh [-z3 true|false] [-print true|false]
 ```
 All parameters are optional. Defaults: `-z3 false`, `-print false`. Passing `z3 True` will use the z3 based generator. Passing `-z3 False` will use the evolutionary algorithm based optimizer. The `-print` flag controls printing detailed output.

 Example:
 ```bash
 (venv) ~/dll-fuzzing-with-input-invariants$ bash run_harness.sh -z3 true -print true
 ```

 To run fuzzing campaings, use the `scripts/fuzz_with_slurm.sh`. **Be sure to install and configure slurm before running this.**. This runs the fuzzing campaign on apis from the file `apis.txt` parallelly.
 ```bash
 (venv) ~/dll-fuzzing-with-input-invariants$ bash scripts/fuzz_with_slurm.sh <duration> <n_max> <lib> <seed>
 ```
 - `duration`: Duration to fuzz each api in seconds.
 - `n_max`: Passing 0 (default) means no max on number of inputs. Anything `> 0` will limit the number of inputs to that number (if it can reach that number before the time budget `duration` runs out).
 - `lib`: `torch` for PyTorch, `tf` for Tenosrflow
 - `seed`: Seed for the generator, default `200`.

 Example:
 ```bash
 (venv) ~/dll-fuzzing-with-input-invariants$ bash scripts/fuzz_with_slurm.sh 3600 0 torch 42
 ```
 This will run the `z3` based generator parallelly on all apis in `apis.txt` with `seed=42`, each with a time budget of 1 hour with no limits on the number of inputs or models generated.