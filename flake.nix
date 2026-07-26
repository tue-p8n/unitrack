{
  description = "Unified multi-object tracking.";

  inputs = {
    tue-p8n.url = "github:tue-p8n/nix";
    nixpkgs.follows = "tue-p8n/nixpkgs";
    treefmt.follows = "tue-p8n/treefmt";
    flake-parts.url = "github:hercules-ci/flake-parts";
    git-hooks = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # Documentation framework
    docyard = {
      url = "github:khwstolle/docyard";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{
      self,
      flake-parts,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.treefmt.flakeModule
        inputs.git-hooks.flakeModule
        inputs.docyard.flakeModules.default
      ];
      systems = [
        "x86_64-linux"
      ];
      perSystem =
        {
          config,
          system,
          pkgs,
          lib,
          ...
        }:
        {
          _module.args.pkgs = import self.inputs.nixpkgs {
            inherit system;
            config = {
              cudaForwardCompat = true;
              cudaSupport = true;
              allowUnfree = true;
            };
          };

          # Formatter.
          treefmt = {
            programs = {
              # Nix
              alejandra.enable = true;
              deadnix.enable = true;

              # Shell
              shellcheck.enable = true;
              shfmt.enable = true;

              # C/C++, CUDA
              clang-format.enable = true;
              clang-tidy.enable = true;

              # Python
              ruff.check = true;
              ruff.format = true;
            };
            settings = {
              formatter = {
                ruff-check.priority = 1;
                ruff-check.options = [ "--fix-only" ];
                ruff-format.priority = 2;
              };
            };
          };

          # Git Hooks.
          # https://github.com/cachix/git-hooks.nix/blob/master/flake-module.nix
          pre-commit.settings = {
            package = pkgs.prek;
            hooks = {
              # Treefmt (see above)
              treefmt = {
                enable = true;
                package = config.treefmt.build.wrapper;
              };

              # File hygiene.
              check-toml.enable = true;
              check-yaml.enable = true;
              check-json.enable = true;
              check-merge-conflicts.enable = true;
              check-added-large-files.enable = true;
              end-of-file-fixer.enable = true;
              trim-trailing-whitespace = {
                enable = true;

                # Preserve markdown "two trailing spaces = line break" semantics.
                args = [ "--markdown-linebreak-ext=md" ];
              };
            };
          };

          # Documentation site (managed mode). `nix run .#docs-serve` to
          # preview, `nix run .#docs-build` to generate the static site.
          # Renders docs/ Markdown plus the `unitrack` Python API via griffe.
          docyard = {
            enable = true;
            site = {
              managed = true;
              title = "Unitrack";
              description = "Unified multi-object tracking.";
              repo = "https://github.com/tue-p8n/unitrack";
              content = "docs";
              # Keep the build output OUT of `content`, otherwise the content
              # collection globs it back in as pages. Also kept out of `dist`,
              # which the Makefile's `dist`/`build` targets use for the
              # Python wheel/sdist (uv build / twine upload dist/*).
              output = "dist-docs";
              # Static files (not Markdown) overlaid onto public/ -- the
              # header logo images live here since the content collection
              # only picks up Markdown.
              assets = "docs-assets";
              logo = {
                light = "/logo-light.svg";
                dark = "/logo-dark.svg";
                alt = "Unitrack";
              };
              favicon = "/favicon.svg";
              apis = [
                {
                  language = "python";
                  name = "unitrack";
                  src = "sources/unitrack";
                }
              ];
              theme = {
                # A cool cyan-teal -- distinct from the theme's default
                # forest green and evocative of the vision/tracking domain.
                primaryHue = 195;
                fonts = {
                  display = "Space Grotesk";
                  sans = "IBM Plex Sans";
                  link = "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap";
                };
              };
            };
          };

          # Packages.
          # TODO

          # Shells for `nix develop.`
          # Provides a CUDA environment.
          devShells = {
            uv-cuda13_0 = inputs.tue-p8n.devShells.${system}.uv-cuda13_0;
            default = self.devShells.${system}.uv-cuda13_0;
          };
        };
      flake = { };
    };
}
