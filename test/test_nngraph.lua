
require 'totem'
require 'nngraph'
local test = {}
local tester = totem.Tester()

local function checkGradients(...)
    totem.nn.checkGradients(tester, ...)
end

function test.test_oneOutput()
    local in1 = nn.Identity()()
    local out1 = nn.Identity()(in1)
    local module = nn.gModule({in1}, {out1})

    local input = torch.Tensor({1})
    module:forward(input)
    tester:eq(module.output, torch.Tensor{1}, "output")
    local gradInput = module:backward(input, torch.Tensor({-123}))
    tester:eq(gradInput, torch.Tensor{-123}, "gradInput")

    local input2 = torch.Tensor({2})
    module:forward(input2)
    tester:eq(module.output, torch.Tensor{2}, "output for input2")
    gradInput = module:backward(input2, torch.Tensor({-2}))
    tester:eq(gradInput, torch.Tensor{-2}, "expecting a recomputed gradInput")
end


function test.test_twoOutputs()
    local in1 = nn.Identity()()
    local out1 = nn.Identity()(in1)
    local out2 = nn.Identity()(in1)
    local module = nn.gModule({in1}, {out1, out2})

    local input = torch.Tensor({1})
    module:forward(input)
    local gradInput = module:backward(input, {torch.Tensor({-2}), torch.Tensor({-3})})
    tester:eq(gradInput, torch.Tensor{-5}, "gradInput of a fork")
    checkGradients(module, input)
end

function test.test_twoGradOutputs()
    local in1 = nn.Sigmoid()()
    local splitTable = nn.SplitTable(1)({in1})
    local out1, out2 = splitTable:split(2)
    local module = nn.gModule({in1}, {out1, out2})

    local input = torch.randn(2, 3)
    local output = module:forward(input)
    assert(#output == 2, "wrong number of outputs")
    module:backward(input, {torch.randn(3), torch.randn(3)})
    checkGradients(module, input)
end

function test.test_twoInputs()
    local in1 = nn.Identity()()
    local in2 = nn.Identity()()
    local prevH, prevCell = in2:split(2)

    local out1 = nn.CMulTable()({in1, prevH, prevCell})
    local module = nn.gModule({in1, in2}, {out1})

    local input = {torch.randn(3), {torch.randn(3), torch.randn(3)}}
    module:forward(input)
    local gradInput = module:backward(input, torch.randn(3))
    assert(#gradInput == 2, "wrong number of gradInputs")
    assert(type(gradInput[2]) == "table", "wrong gradInput[2] type")
    checkGradients(module, input)
end

function test.test_twoInputs2()
    local in1 = nn.Sigmoid()()
    local in2 = nn.Sigmoid()()
    local module = nn.gModule({in1, in2}, {in1, in2, nn.Sigmoid()(in1)})

    local input = {torch.randn(3), torch.randn(3)}
    module:forward(input)
    local gradInput = module:backward(input, {torch.randn(3), torch.randn(3), torch.randn(3)})
    checkGradients(module, input)
end

function test.test_splitDebugLabels()
    local node = nn.Identity()()
    node.data.annotations._debugLabel = "node"
    local node1, node2 = node:split(2)
    assert(node1.data.annotations._debugLabel == "node-1")
    assert(node2.data.annotations._debugLabel == "node-2")
end

function test.test_identity()
    local in1 = nn.Identity()()
    local in2 = nn.Identity()()
    local module = nn.gModule({in1, in2}, {in1, in2, nn.Identity()(in1)})

    local input = {torch.randn(3), torch.randn(3)}
    module:forward(input)
    module:backward(input, {torch.randn(3), torch.randn(3), torch.randn(3)})
    checkGradients(module, input)
end

function test.test_gradInputType()
    local xInput = torch.randn(3)
    local h = torch.randn(3)

    local x = nn.Identity()()
    local prevRnnState = nn.Identity()()
    local prevH1, prevCell = prevRnnState:split(2)
    local prevH = prevH1

    local cellOut = nn.CAddTable()({
            nn.CMulTable()({x, prevH}),
            nn.CMulTable()({prevH, prevCell})})
    local module = nn.gModule({x, prevRnnState}, {cellOut})

    local c = torch.randn(h:size())
    local prevRnnState = {h, c}
    local input = {xInput, prevRnnState}
    local output = module:forward(input)

    local gradOutput = torch.randn(h:size())
    local gradInput = module:backward(input, gradOutput)

    local gradX, gradPrevState = unpack(gradInput)
    local gradPrevH, gradPrevCell = unpack(gradPrevState)
    assert(type(gradPrevH) == type(h), "wrong gradPrevH type")

    tester:eq(type(gradPrevH), type(h), "wrong gradPrevH type")
    tester:eq(gradPrevH:size(), h:size(), "wrong gradPrevH size")
    checkGradients(module, input)
end

function test.test_tabularInput()
    local in1 = nn.SplitTable(1)()
    local out1 = nn.CAddTable()(in1)
    local module = nn.gModule({in1}, {out1})

    local input = torch.randn(2, 3)
    checkGradients(module, input)
end

function test.test_extraTable()
    local in1 = nn.Identity()()
    local out1 = nn.Identity()(in1)
    local module = nn.gModule({in1}, {out1})

    local input = torch.Tensor({123})
    tester:eq(module:forward(input), input, "simple output")
    tester:eq(module:forward({input}), {input}, "tabular output")
end

function test.test_accGradParameters()
    local input = torch.randn(10)

    local in1 = nn.CMul(input:nElement())()
    local out1 = nn.Identity()(in1)
    local out2 = nn.Identity()(in1)
    local module = nn.gModule({in1}, {out1, out2})
    checkGradients(module, input)
end

function test.test_example1()
    local x1 = nn.Linear(20,10)()
    local mout = nn.Linear(10,1)(nn.Tanh()(nn.Linear(10,10)(nn.Tanh()(x1))))
    local mlp = nn.gModule({x1},{mout})

    local x = torch.rand(20)
    checkGradients(mlp, x)
end

function test.test_example2()
    local x1=nn.Linear(20,20)()
    local x2=nn.Linear(10,10)()
    local m0=nn.Linear(20,1)(nn.Tanh()(x1))
    local m1=nn.Linear(10,1)(nn.Tanh()(x2))
    local madd=nn.CAddTable()({m0,m1})
    local m2=nn.Sigmoid()(madd)
    local m3=nn.Tanh()(madd)
    local gmod = nn.gModule({x1,x2},{m2,m3})

    local x = torch.rand(20)
    local y = torch.rand(10)
    checkGradients(gmod, {x, y})
end

function test.test_example3()
    local m = nn.Sequential()
    m:add(nn.SplitTable(1))
    m:add(nn.ParallelTable():add(nn.Linear(10,20)):add(nn.Linear(10,30)))
    local input = nn.Identity()()
    local input1,input2 = m(input):split(2)
    local m3 = nn.JoinTable(1)({input1,input2})
    local g = nn.gModule({input},{m3})

    local indata = torch.rand(2,10)
    checkGradients(g, indata)
end

function test.test_example4()
    local input = nn.Identity()()
    local L1 = nn.Tanh()(nn.Linear(1,2)(input))
    local L2 = nn.Tanh()(nn.Linear(3,6)(nn.JoinTable(1)({input,L1})))
    local L3 = nn.Tanh()(nn.Linear(8,16)(nn.JoinTable(1)({L1,L2})))
    local g = nn.gModule({input},{L3})

    local indata = torch.rand(1)
    checkGradients(g, indata)
end

function test.test_type()
    local in1 = nn.Linear(20,10)()
    local out1 = nn.Linear(10,1)(nn.Tanh()(nn.Linear(10,10)(nn.Tanh()(in1))))
    local module = nn.gModule({in1}, {out1})
    local input = torch.rand(20)
    local output = module:forward(input)
    module:backward(input, output)
    tester:eq(torch.typename(output), "torch.DoubleTensor")
    tester:eq(torch.typename(module.output), "torch.DoubleTensor")
    tester:eq(torch.typename(module.gradInput), "torch.DoubleTensor")
    tester:eq(torch.typename(module.innode.data.input[1]), "torch.DoubleTensor")
    tester:eq(torch.typename(module.outnode.data.input[1]), "torch.DoubleTensor")
    tester:eq(torch.typename(module.forwardnodes[1].data.input[1]), "torch.DoubleTensor")

    module:float()
    local output = module:forward(input:float())
    tester:eq(torch.typename(output), "torch.FloatTensor")
    tester:eq(torch.typename(module.output), "torch.FloatTensor")
    tester:eq(torch.typename(module.gradInput), "torch.FloatTensor")
    tester:eq(torch.typename(module.innode.data.input[1]), "torch.FloatTensor")
    tester:eq(torch.typename(module.outnode.data.input[1]), "torch.FloatTensor")
    tester:eq(torch.typename(module.forwardnodes[1].data.input[1]), "torch.FloatTensor")
end

function test.test_nestedGradInput()
    local x = nn.Identity()()
    local h1 = nn.Sequential():add(nn.JoinTable(2)):add(nn.Tanh())
    local h2 = nn.Sequential():add(nn.JoinTable(2)):add(nn.Identity())
    local out = nn.CAddTable()({h1(x), h2(x)})

    local model = nn.gModule({x}, {out})

    local input = {}
    input[1] = torch.randn(3, 3)
    input[2] = torch.randn(3, 3)
    input[3] = torch.randn(3, 3)

    checkGradients(model, input)

    local input = {}
    input[1] = torch.randn(2, 3)
    input[2] = torch.randn(2, 3)
    input[3] = torch.randn(2, 3)

    checkGradients(model, input)
end

function test.test_unusedInput()
    local x = nn.Identity()()
    local h = nn.Identity()()
    local h2 = nn.Identity()()

    local ok, result = pcall(nn.gModule, {x, h}, {x})
    assert(not ok, "the unused input should be detected")
end

function test.test_unusedChild()
    local prevState = nn.Identity()()
    local h, cell = prevState:split(2)

    local ok, result = pcall(nn.gModule, {prevState}, {h})
    assert(not ok, "the unused cell should be detected")
end

function test.test_nilInput()
    local ok, result = pcall(function() nn.Sigmoid()(nil) end)
    assert(not ok, "the nil input should be detected")
end

function test.test_unusedNode()
    local in1 = nn.Identity()()
    local in2 = nn.Identity()()
    local middleResult = nn.Sigmoid()(in2)
    local out1 = nn.Sigmoid()(in1)

    local ok, result = pcall(nn.gModule, {in1, in2}, {out1})
    assert(not ok, "the unused middleResult should be detected")
end

function test.test_usageAfterSplit()
    local prevState = nn.Identity()()
    local h, cell = prevState:split(2)
    local nextState = nn.Identity()(prevState)
    local transformed = nn.Sigmoid()(cell)

    local model = nn.gModule({prevState}, {h, nextState, transformed})
    local nHidden = 10
    local input = {torch.randn(nHidden), torch.randn(nHidden)}
    checkGradients(model, input)
end

function test.test_resizeNestedAs()
    local in1 = nn.Identity()()
    local out1 = nn.Identity()(in1)
    local out2 = nn.Identity()(in1)

    local net = nn.gModule({in1}, {out1, out2})
    local input = {torch.randn(10), {torch.randn(3), torch.randn(4)}}
    net:forward(input)
    net:backward(input, net.output)
    checkGradients(net, input)

    input = {torch.randn(10), {torch.randn(3), torch.randn(4), torch.randn(5)}}
    net:forward(input)
    net:backward(input, net.output)
    checkGradients(net, input)

    input = {torch.randn(10), {torch.randn(3), torch.randn(4)}}
    net:forward(input)
    local gradInput = net:backward(input, net.output)
    tester:eq(#(gradInput[2]), 2, "gradInput[2] size")
    checkGradients(net, input)
end


function test.test_annotateGraph()
  local input = nn.Identity()():annotate(
      {name = 'Input', description = 'DescA',
       graphAttributes = {color = 'red'}})

  local hidden_a = nn.Linear(10, 10)(input):annotate(
      {name = 'Hidden A', description = 'DescB',
       graphAttributes = {color = 'blue', fontcolor='green', tooltip = 'I am green'}})
  local hidden_b = nn.Sigmoid()(hidden_a)
  local output = nn.Linear(10, 10)(hidden_b)
  local net = nn.gModule({input}, {output})

  tester:assert(hidden_a:label():match('DescB'))
  local fg_tmpfile = os.tmpname()
  local bg_tmpfile = os.tmpname()
  graph.dot(net.fg, 'Test', fg_tmpfile)
  graph.dot(net.fg, 'Test BG', bg_tmpfile)

  local function checkDotFile(tmpfile)
    local dotcontent = io.open(tmpfile .. '.dot', 'r'):read("*all")
    tester:assert(
        dotcontent:match('%[color=red.*label=%"Input.*DescA.*%".*%]'))
    tester:assert(
        dotcontent:match(
          '%[.*fontcolor=green.*label=%"Hidden A.*DescB.*%".*%]'))
    tester:assert(
        dotcontent:match('%[color=blue.*label=%".*DescB.*%".*%]'))
    tester:assert(
        dotcontent:match(
          '%[.*label=%".*DescB.*%".*tooltip=%"I am green%".*%]'))
  end

  checkDotFile(fg_tmpfile)
  checkDotFile(bg_tmpfile)
end


tester:add(test):run()
