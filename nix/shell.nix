{...}: {
  perSystem = {pkgs, ...}: {
    devShells.default = pkgs.mkShell {
      packages = with pkgs; [
        (rust-bin.fromRustupToolchainFile ../rust-toolchain.toml)
        cargo-deny
        cargo-insta
        nodejs-slim
        corepack
        bun
        wasm-bindgen-cli_0_2_127
      ];
      shellHook = ''
        corepack install
        export PATH="$PATH:$(pwd)/node_modules/.bin"
      '';
    };
  };
}
