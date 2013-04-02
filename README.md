
# torch-nngraph

This package provides graphical computation for nn library in Torch7.

## Requirements

### torch-graph

This library requires torch-graph package to be installed.

[[http://github.com/koraykv/torch-graph]]

### graphviz

You do *not* need graphviz to be able to use this library, but if you have one then you can display the graphs that you have created.

## Installation

Right now, this repo is not distributed as part of torch-pkg or luarocks system. For installation follow these steps.

git clone git://github.com/koraykv/torch-nngraph.git
cd torch-nngraph
torch-pkg deploy 


## Usage

The aim of this library is to provide users of nn library with tools to easily create complicated architectures. Any given nn module or criterion is going to be bundled into a graph node. The __call operator of an instance of nn.Module and nn.Criterion is used to create architectures as if one is writing function calls.

### One hidden layer network.

```lua
require 'nngraph'

mout = nn.Linear(10,1)(nn.Tanh()(nn.Linear(20,10)()))
mlp = nn.gModule(mout)

x = torch.rand(20)
dx = torch.rand(1)
mlp:updateOutput(x)
mlp:updateGradInput(x,dx)
mlp:accGradParameters(x,dx)

-- draw graph
graph.dot(mlp.fg)


```

![mlp](https://raw.github.com/koraykv/torch-nngraph/master/doc/mlp.svg)