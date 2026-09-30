{...}: {
  perSystem = {pkgs, ...}: {
    devShells.default = pkgs.mkShell {
      packages = with pkgs; [
        (rust-bin.fromRustupToolchainFile ../rust-toolchain.toml)
        cargo-deny
        cargo-insta
        nodejs
        bun
        wasm-bindgen-cli_0_2_127
      ];
    };
  };
}
