package = "nngraph"
version = "scm-1"

source = {
   url = "git://github.com/torch/nngraph",
   tag = "master"
}

description = {
   summary = "This package provides graphical computation for nn library in Torch7.",
   homepage = "https://github.com/torch/nngraph",
   license = "UNKNOWN"
}

dependencies = {
   "torch >= 7.0",
   "graph",
   "nn"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}
