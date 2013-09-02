
CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)

SET(luasrc init.lua gmodule.lua node.lua utils.lua nesting.lua
    test.lua)

ADD_TORCH_PACKAGE(nngraph ""  "${luasrc}" "Neural Net Graph Package")
