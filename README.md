# BMS NLP Benchmark
A benchmark to test how well NLP models understand BMS point names across different buildings. Includes datasets and baseline experiments.

This benchmark dataset is derived from the BMS dataset published by [Fierro _et al_](https://doi.org/10.5281/zenodo.3455825).


---

## Setup Instructions

### 1. Install Poetry

[Poetry](https://python-poetry.org/docs/#installation) manages Python dependencies and virtual environments for this project. If Poetry is not already installed, it can be installed with:

```bash
pip install poetry
```

### 2. Install Project Dependencies

Clone this repository and install dependencies using Poetry:

```bash
poetry install
```

---

### 3. Testing and Other Dev Commands

* Installing pre-commit hooks (to automatically run code quality checks locally on git push, install once)

   * Install the Git hooks (via Poetry so the correct venv is used):
   ```bash
   poetry run pre-commit install              # installs the pre-commit hook
   poetry run pre-commit install --hook-type pre-push   # for the pytest hook
   ```
   * First run on the whole repo (recommended):
   ```bash
   poetry run pre-commit run --all-files
   # and to simulate the pre-push stage once:
   poetry run pre-commit run --all-files --hook-stage push
   ```

* Running pytest integration tests:

```bash
poetry run pytest -v -m integration
```

* Running pytest unit test coverage report:

```bash
poetry run pytest --cov-config=.coveragerc --cov=./ tests/
```
* Run the full test suite in isolated environments including code quality tests:
```bash
poetry run tox
```
* If package dependencies have changed since the last run of `tox`, run
```bash
poetry run tox -r
```



---
## References:
[Fierro _et al_] Fierro, G., Guduguntla, S., & Culler, D. E. (2019). Dataset: An Open Dataset and Collection Tool for BMS Point Labels [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3455825

