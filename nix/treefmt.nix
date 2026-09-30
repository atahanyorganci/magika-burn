{
  perSystem = {...}: {
    treefmt = {
      projectRootFile = "flake.nix";
      programs = {
        deadnix.enable = true;
        alejandra.enable = true;
        mdsh.enable = true;
      };
    };
  };
}
