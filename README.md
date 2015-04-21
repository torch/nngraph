# Neural Network Graph Package

This package provides graphical computation for `nn` library in [Torch](https://github.com/torch/torch7/blob/master/README.md).

## Requirements

You do *not* need graphviz to be able to use this library, but if you have then you can display the graphs that you have created.

## Usage

[Plug: A more explanatory nngraph tutorial by Nando De Freitas of  Oxford](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/practicals/practical5.pdf)

The aim of this library is to provide users of nn library with tools to easily create complicated architectures.
Any given nn module is going to be bundled into a graph node.
The `__call` operator of an instance of `nn.Module` is used to create architectures as if one is writing function calls.

### One hidden layer network

```lua
require 'nngraph'

x1 = nn.Linear(20,10)()
mout = nn.Linear(10,1)(nn.Tanh()(nn.Linear(10,10)(nn.Tanh()(x1))))
mlp = nn.gModule({x1},{mout})

x = torch.rand(20)
dx = torch.rand(1)
mlp:updateOutput(x)
mlp:updateGradInput(x, dx)
mlp:accGradParameters(x, dx)

-- draw graph (the forward graph, '.fg')
graph.dot(mlp.fg, 'MLP')
```

<img src= "https://raw.github.com/koraykv/torch-nngraph/master/doc/mlp.png" width="300px"/>

Read this diagram from top to bottom, with the first and last nodes being dummy nodes that regroup all inputs and outputs of the graph.
The 'module' entry describes the function of the node, as applies to 'input', and producing an result of the shape 'gradOutput'; 'mapindex' contains pointers to the parent nodes.


### A net with 2 inputs and 2 outputs

```lua
require 'nngraph'

x1 = nn.Linear(20, 20)()
x2 = nn.Linear(10, 10)()
m0 = nn.Linear(20, 1)(nn.Tanh()(x1))
m1 = nn.Linear(10, 1)(nn.Tanh()(x2))
madd = nn.CAddTable()({m0, m1})
m2 = nn.Sigmoid()(madd)
m3 = nn.Tanh()(madd)
gmod = nn.gModule({x1, x2}, {m2, m3})

x = torch.rand(20)
y = torch.rand(10)

gmod:updateOutput({x, y})
gmod:updateGradInput({x, y}, {torch.rand(1), torch.rand(1)})
graph.dot(gmod.fg, 'Big MLP')
```

<img src= "https://raw.github.com/koraykv/torch-nngraph/master/doc/mlp2.png" width="300px"/>


### Another net that uses container modules (like `ParallelTable`) that output a table of outputs

```lua
m = nn.Sequential()
m:add(nn.SplitTable(1))
m:add(nn.ParallelTable():add(nn.Linear(10, 20)):add(nn.Linear(10, 30)))
input = nn.Identity()()
input1,input2 = m(input):split(2)
m3 = nn.JoinTable(1)({input1, input2})

g = nn.gModule({input}, {m3})

indata = torch.rand(2, 10)
gdata = torch.rand(50)
g:forward(indata)
g:backward(indata, gdata)

graph.dot(g.fg, 'Forward Graph')
graph.dot(g.bg, 'Backward Graph')
```

<img src= "https://raw.github.com/koraykv/torch-nngraph/master/doc/mlp3_forward.png" width="300px"/>
<img src= "https://raw.github.com/koraykv/torch-nngraph/master/doc/mlp3_backward.png" width="300px"/>


### A Multi-layer network where each layer takes output of previous two layers as input

```lua
input = nn.Identity()()
L1 = nn.Tanh()(nn.Linear(10, 20)(input))
L2 = nn.Tanh()(nn.Linear(30, 60)(nn.JoinTable(1)({input, L1})))
L3 = nn.Tanh()(nn.Linear(80, 160)(nn.JoinTable(1)({L1, L2})))

g = nn.gModule({input}, {L3})

indata = torch.rand(10)
gdata = torch.rand(160)
g:forward(indata)
g:backward(indata, gdata)

graph.dot(g.fg, 'Forward Graph')
graph.dot(g.bg, 'Backward Graph')
```

<img src= "https://raw.github.com/koraykv/torch-nngraph/master/doc/mlp4_forward.png" width="300px"/>
<img src= "https://raw.github.com/koraykv/torch-nngraph/master/doc/mlp4_backward.png" width="300px"/>


## Annotations

It is possible to add annotations to your network, such as labeling nodes with names or attributes which will show up when you graph the network.
This can be helpful in large graphs.

For the full list of graph attributes see the
[graphviz documentation](http://www.graphviz.org/doc/info/attrs.html).

```lua
input = nn.Identity()()
L1 = nn.Tanh()(nn.Linear(10, 20)(input)):annotate{
   name = 'L1', description = 'Level 1 Node',
   graphAttributes = {color = 'red'}
}
L2 = nn.Tanh()(nn.Linear(30, 60)(nn.JoinTable(1)({input, L1}))):annotate{
   name = 'L2', description = 'Level 2 Node',
   graphAttributes = {color = 'blue', fontcolor = 'green'}
}
L3 = nn.Tanh()(nn.Linear(80, 160)(nn.JoinTable(1)({L1, L2}))):annotate{
   name = 'L3', descrption = 'Level 3 Node',
   graphAttributes = {color = 'green',
   style='filled', fillcolor = 'yellow'}
}

g = nn.gModule({input},{L3})

indata = torch.rand(10)
gdata = torch.rand(160)
g:forward(indata)
g:backward(indata, gdata)

graph.dot(g.fg,'Forward Graph', '/tmp/fg')
graph.dot(g.bg,'Backward Graph', '/tmp/bg')
```

![Annotated forward graph](doc/annotation_fg.png?raw=true)
![Annotated backward graph](doc/annotation_bg.png?raw=true)
