{pkgs, ...}: {
  # https://devenv.sh/reference/options/
  packages = with pkgs; [
    git
    lazygit
    just
    python313
    uv
    pdm
    ruff
  ];

  enterShell = ''
    just install
  '';

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
    # trailing-whitespace.enable = true;
    no-commit-to-branch.enable = true;

    # execute example shell from Markdown files
    mdsh.enable = true;
  };
}
