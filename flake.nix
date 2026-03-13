{
  description = "UniTrack";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      perSystem =
        {
          config,
          self',
          inputs',
          pkgs,
          system,
          ...
        }:
        {
          devShells.default =
            let
              systemLibs = with pkgs; [
                stdenv.cc.cc.lib # libstdc++.so.6
                zlib # libz.so
                glib # libglib-2.0.so
                libxkbcommon # Common requirement for many wheels
                libglvnd
                libGL
              ];
            in
            pkgs.mkShell {
              name = "unitrack";
              packages = with pkgs; [
                uv
                git
                pkg-config
                ninja
                which
                cacert
                just
              ];
              env = {
                # Force uv to copy libs so they don't rely on hardlinks (safer in Nix shells)
                UV_LINK_MODE = "copy";

                UV_PYTHON_PREFERENCE = "only-managed";
                UV_PYTHON_DOWNLOADS = "auto";
                UV_NO_BUILD_ISOLATION = true;
                UV_TORCH_BACKEND = "auto";

                # CA certificates
                SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
                NIX_SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";

                # NIX_LD setup to allow unpatched wheels to run
                NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath systemLibs;
                NIX_LD = pkgs.lib.fileContents "${pkgs.stdenv.cc}/nix-support/dynamic-linker";
              };

              shellHook = ''
                unset PYTHONPATH

                export REPO_ROOT=$(git rev-parse --show-toplevel)
                export UV_CACHE_DIR="$REPO_ROOT/.uv_cache"
                export UV_PROJECT_ENVIRONMENT="$REPO_ROOT/.venv"
              '';
            };
        };
    };
}
