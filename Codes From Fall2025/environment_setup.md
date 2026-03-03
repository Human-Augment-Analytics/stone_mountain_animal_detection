# Environment Setup

Follow these steps to create a reproducible Python environment, install the repository requirements, and bring in the latest `diffusers` package from source.

## 1. Create and Activate a Virtual Environment

```bash
python -m venv animal_env
source animal_env/bin/activate        # Linux/macOS
animal_env\Scripts\activate           # Windows (PowerShell)
```

Check that the environment is active:

```bash
python -c "import sys; print(sys.executable)"
# -> .../HAAG_Stone_Mountain_Species_Detection_Project/animal_env/bin/python
```

## 2. Install Repository Requirements

With the virtual environment active, install the dependencies listed in `requirements.txt`:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

You can confirm the installed packages at any time:

```bash
pip list
```

## 3. Install Hugging Face Diffusers from Source

The `diffusers` dependency is installed directly from the Hugging Face repository so that you can track the latest changes.

```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e ".[torch,flax,tensorflow,quality,test]"   # choose extras that match your stack
```

If you already have a specific CUDA-enabled PyTorch build, install it before running the command above, and drop the `torch` extra to prevent version changes.

Later updates only require pulling the repository—the editable install picks up changes automatically:

```bash
cd diffusers
git pull
```

## 4. Keeping the Environment Clean

- Reactivate `animal_env` (`source animal_env/bin/activate`) whenever you return to the project.
- Use `pip freeze > requirements.lock` if you want to snapshot the exact package versions in use.
- When finished working, deactivate the environment with `deactivate`.
