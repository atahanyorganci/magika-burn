{inputs, ...}: {
  systems = [
    "aarch64-darwin"
    "aarch64-linux"
    "x86_64-linux"
  ];
  imports = [
    inputs.treefmt-nix.flakeModule
  ];
  perSystem = {system, ...}: let
    pkgs = import inputs.nixpkgs {
      inherit system;
      overlays = [inputs.rust-overlay.overlays.default];
      config = {
        allowUnfree = true;
        allowBroken = true;
      };
    };
  in {
    _module.args = {
      inherit pkgs;
    };
  };
}
