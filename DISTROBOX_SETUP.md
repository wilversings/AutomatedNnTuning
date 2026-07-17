# Running in a distrobox (2026 revival)

The code targets the 2018-era Keras/TensorFlow stack (Keras 2.1.4 API: `keras.layers.core`,
`weights=` kwarg on `Dense`, `keras.utils.vis_utils.plot_model`), which needs Python 3.6 and
TensorFlow 1.5. No modern distro ships Python 3.6, so the setup uses a distrobox with a
micromamba-managed Python 3.6 environment inside it.

The original `requirements.txt` files are full system `pip freeze` dumps; the actual
dependencies are the short list below.

## Distrobox

```sh
distrobox create --name nn-thesis --image registry.fedoraproject.org/fedora-toolbox:latest --yes
```

## Dependencies installed inside the distrobox

System package (via `sudo dnf install`):

- `graphviz` (the `dot` binary, needed by `keras.utils.vis_utils.plot_model`)

Tooling:

- `micromamba` (static binary, installed to `~/.local/share/nn-thesis/bin/micromamba`,
  root prefix `~/.local/share/nn-thesis/mamba`)

Conda env `thesis` (from conda-forge):

- `python=3.6` (resolved to 3.6.15)
- `pip`

Pip packages inside the `thesis` env:

- `tensorflow==1.5.0` (CPU build; the original used `tensorflow-gpu==1.5.0`)
- `keras==2.1.4`
- `numpy==1.14.2`
- `scipy==1.0.0`
- `scikit-learn==0.19.2`
- `h5py==2.8.0`
- `sortedcontainers==1.5.9`
- `pydot==1.2.4`

## Setup commands

```sh
distrobox enter nn-thesis

# inside the box:
sudo dnf install -y graphviz
mkdir -p ~/.local/share/nn-thesis/bin && cd ~/.local/share/nn-thesis
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
export MAMBA_ROOT_PREFIX=~/.local/share/nn-thesis/mamba
./bin/micromamba create -y -n thesis -c conda-forge python=3.6 pip
./bin/micromamba run -n thesis pip install \
    tensorflow==1.5.0 keras==2.1.4 numpy==1.14.2 scipy==1.0.0 \
    scikit-learn==0.19.2 h5py==2.8.0 sortedcontainers==1.5.9 pydot==1.2.4
```

## Running

```sh
distrobox enter nn-thesis
export MAMBA_ROOT_PREFIX=~/.local/share/nn-thesis/mamba
cd Console
PYTHONPATH=.. ~/.local/share/nn-thesis/bin/micromamba run -n thesis python entrypoint.py
```

Each run creates a `Console/<unix-timestamp>/` directory containing `general.log` plus the
best model per generation (`model_genN.json`, `model_genN.h5`, `model_genN.png`).
