# scratch

## Installation

### Nix

If you do not know what nix is or don't have it installed please look at
https://devenv.sh/getting-started/

```bash
devenv init # initialize a development environment
devenv shell # start a development shell with you centered
```

If you have NixOS with flake features

```bash
nix develop
```

Once your shell is created and available install the project with the following
command

```bash
just install
```

For more task commands

```bash
just help
```

### PDM

This project uses UV as it back-end project manager, but you can easily make it
work with PDM [pdm with uv backend](https://pdm-project.org/en/latest/usage/uv/)
This shows that you can configure PDM to use UV as it's backend. Thus you are
able run the following, if you must use PDM

```bash
pdm config use_uv true
pdm config python.install_root $(uv python dir --color never)
pdm install
```
