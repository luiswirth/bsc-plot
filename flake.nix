{
  description = "plot";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
      };
    in
    {
      formatter.${system} = pkgs.nixpkgs-fmt;

      devShell.${system} = pkgs.mkShell {
        nativeBuildInputs = with pkgs; [

          python312
          python312Packages.ipython
          python312Packages.python-lsp-server
          python312Packages.pyqt6

          glib
          zlib
          libGL
          fontconfig
          xorg.libX11
          libxkbcommon
          freetype
          dbus
          
        ];
      };
    };
}
