{pkgs, ...}: {
  # https://devenv.sh/reference/options/
  packages = with pkgs;
    [
      # project essentials packages
      git
      netcat
      lazygit
      just
      ruff
    ]
    ++ [
      # packages for local GitHub actions
      act
      docker
      procps
    ];

  enterShell = ''
    # Create a local symlink to the Python virtual environment
    if [ ! -L "$DEVENV_ROOT/.venv" ]; then
        ln -s "$DEVENV_STATE/venv/" "$DEVENV_ROOT/.venv"
    fi
  '';

  enterTest = ''
    just --version | grep "${pkgs.just.version}"
    uv --version | grep "${pkgs.uv.version}"
    python --version | grep "${pkgs.python313.version}"
  '';

  processes = {
    server.exec = "just api";
  };

  git-hooks.hooks = {
    # Nix
    alejandra.enable = true;
    deadnix.enable = true;

    # Python
    ruff = {
      enable = true;
      args = ["--unsafe-fixes" "--exit-non-zero-on-fix"];
    };
    ruff-format.enable = true;
    uv-check.enable = true;
    uv-lock.enable = true;

    # Data files
    check-toml.enable = true;
    check-json.enable = true;
    pretty-format-json = {
      enable = true;
      args = ["--autofix" "--no-sort-keys"];
    };

    # Misc checks
    check-added-large-files.enable = true;
    check-case-conflicts.enable = true;
    check-merge-conflicts.enable = true;

    # Global hooks
    end-of-file-fixer.enable = true;
    trim-trailing-whitespace.enable = true;
    no-commit-to-branch.enable = true;

    # execute example shell from Markdown files
    mdsh.enable = true;
  };

  languages.python = {
    enable = true;
    uv = {
      enable = true;
      sync = {
        enable = true;
        allPackages = true;
        arguments = ["--frozen"];
      };
    };
  };
}
