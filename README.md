# torch-nngraph

This package provides graphical computation for nn library in Torch7.

## Requirements

### torch-graph

This library requires torch-graph package to be installed.

http://github.com/koraykv/torch-graph

### graphviz

You do *not* need graphviz to be able to use this library, but if you have then you can display the graphs that you have created.

## Installation

Right now, this repo is not distributed as part of torch-pkg or luarocks system. For installation follow these steps.

```
	git clone git://github.com/koraykv/torch-graph.git
	cd torch-graph
	torch-pkg deploy 
	cd ..
	git clone git://github.com/koraykv/torch-nngraph.git
	cd torch-nngraph
	torch-pkg deploy 
```
## Usage

The aim of this library is to provide users of nn library with tools to easily create complicated architectures. Any given nn module is going to be bundled into a graph node. The __call operator of an instance of nn.Module is used to create architectures as if one is writing function calls.

### One hidden layer network

```lua
require 'nngraph'

x1 = nn.Linear(20,10)()
mout = nn.Linear(10,1)(nn.Tanh()(nn.Linear(10,10)(nn.Tanh()(x1))))
mlp = nn.gModule({x1},{mout})

x = torch.rand(20)
dx = torch.rand(1)
mlp:updateOutput(x)
mlp:updateGradInput(x,dx)
mlp:accGradParameters(x,dx)

-- draw graph (the forward graph, '.fg')
graph.dot(mlp.fg,'MLP')


```

<img src= "https://raw.github.com/koraykv/torch-nngraph/master/doc/mlp.png" width="300px"/>
<!-- ![mlp](https://raw.github.com/koraykv/torch-nngraph/master/doc/mlp.png) -->

Read this diagram from top to bottom, with the first and last nodes being dummy nodes that regroup all inputs and outputs of the graph.
The 'module' entry describes the function of the node, as applies to 'input', and producing an result of the shape 'gradOutput'; 'mapindex' contains
pointers to the parent nodes. 



### A net with 2 inputs and 2 outputs

```lua
require 'nngraph'

x1=nn.Linear(20,20)()
x2=nn.Linear(10,10)()
m0=nn.Linear(20,1)(nn.Tanh()(x1))
m1=nn.Linear(10,1)(nn.Tanh()(x2))
madd=nn.CAddTable()({m0,m1})
m2=nn.Sigmoid()(madd)
m3=nn.Tanh()(madd)
gmod = nn.gModule({x1,x2},{m2,m3})

x = torch.rand(20)
y = torch.rand(10)

gmod:updateOutput({x,y})
gmod:updateGradInput({x,y},{torch.rand(1),torch.rand(1)})
graph.dot(gmod.fg,'Big MLP')

```

<img src= "https://raw.github.com/koraykv/torch-nngraph/master/doc/mlp2.png" width="300px"/>

### Another net that uses container modules (like `ParallelTable`) that output a table of outputs.

```lua
	m = nn.Sequential()
	m:add(nn.SplitTable(1))
	m:add(nn.ParallelTable():add(nn.Linear(10,20)):add(nn.Linear(10,30)))
	input = nn.Identity()()
	input1,input2 = m(input):split(2)
	m3 = nn.JoinTable(1)({input1,input2})

	g = nn.gModule({input},{m3})

	indata = torch.rand(2,10)
	gdata = torch.rand(50)
	g:forward(indata)
	g:backward(indata,gdata)

	graph.dot(g.fg,'Forward Graph')
	graph.dot(g.bg,'Backward Graph')
```
<img src= "https://raw.github.com/koraykv/torch-nngraph/master/doc/mlp3_forward.png" width="300px"/>
<img src= "https://raw.github.com/koraykv/torch-nngraph/master/doc/mlp3_backward.png" width="300px"/>

### A Multi-layer network where each layer takes output of previous two layers as input.
```lua
	input = nn.Identity()()
	L1 = nn.Tanh()(nn.Linear(10,20)(input))
	L2 = nn.Tanh()(nn.Linear(30,60)(nn.JoinTable(1)({input,L1})))
	L3 = nn.Tanh()(nn.Linear(80,160)(nn.JoinTable(1)({L1,L2})))

	g = nn.gModule({input},{L3})

	indata = torch.rand(10)
	gdata = torch.rand(160)
	g:forward(indata)
	g:backward(indata,gdata)

	graph.dot(g.fg,'Forward Graph')
	graph.dot(g.bg,'Backward Graph')
```

<img src= "https://raw.github.com/koraykv/torch-nngraph/master/doc/mlp4_forward.png" width="300px"/>
<img src= "https://raw.github.com/koraykv/torch-nngraph/master/doc/mlp4_backward.png" width="300px"/>


