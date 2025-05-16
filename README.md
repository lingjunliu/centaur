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

<h1>Use cases</h1>

<h2> 1. Learn invariants </h2>

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

<h2> 2. Generate inputs </h2>

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
 (venv) ~/dll-fuzzing-with-input-invariants$ bash scripts/fuzz_with_slurm.sh <duration> <mode> <n_max> <limit> <seed>
 ```
 - `Duration`: Duration to fuzz each api in seconds.
 - `Mode`: `z3` for Z3 based generator, `ea` for evolutionary algorithm based optimizer
 - `n_max`: Passing 0 (default) means no max on number of inputs. Anything `> 0` will limit the number of inputs to that number (if it can reach that number before the time budget `duration` runs out). If `n_max > 0` is passed, for `z3` this will also limit the number of models initially generated.
 - `limit`: Deafult `30`
    - For `z3`, this limit represents the percentage of the total `duration` spent on initial model generation
    - For `ea`, this limit represents the duration after which a random restart will take place.
 - `seed`: Seed for the generator, default `200`

 Example:
 ```bash
 (venv) ~/dll-fuzzing-with-input-invariants$ bash scripts/fuzz_with_slurm.sh 3600 z3 0 50 42
 ```
 This will run the `z3` based generator parallelly on all apis in `apis.txt` with `seed=42`, each with a time budget of 1 hour with no limits on the number of inputs or models generated. 50% of this 1 hour i.e. 30 minutes will be spent on model generation, the rest of the time will be spent on input generation (sampling from the valid models and concretizing the inputs).