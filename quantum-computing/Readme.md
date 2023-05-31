For active develpment we are using

* Python 3.8
* [Pyenv][3]
* [Poetry][4]

```bash
git clone git@github.com:PCPUNMSM/quantum-computing.git
cd quantum-computing
```

- Install dependencies

```bash
poetry install
poetry run jupyter labextension install \
    @jupyter-widgets/jupyterlab-manager \
    @jupyterlab/debugger \
    @jupyterlab/git \
    @jupyterlab/toc \
    @krassowski/jupyterlab_go_to_definition \
    @lckr/jupyterlab_variableinspector \
    @ryantam626/jupyterlab_code_formatter \
    nbdime-jupyterlab
```

- Enjoy the notebooks :D

```bash
poetry run jupyter lab
```
