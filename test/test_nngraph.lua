
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

      local unpack = unpack or table.unpack
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
      local gradOutput = output:clone():normal()
      local gradInput = module:backward(input, gradOutput)

      module:backward(input, output)
      tester:eq(torch.typename(output), "torch.DoubleTensor")
      tester:eq(torch.typename(module.output), "torch.DoubleTensor")
      tester:eq(torch.typename(module.gradInput), "torch.DoubleTensor")
      tester:eq(torch.typename(module.innode.data.input[1]), "torch.DoubleTensor")
      tester:eq(torch.typename(module.outnode.data.input[1]), "torch.DoubleTensor")
      tester:eq(torch.typename(module.forwardnodes[1].data.input[1]), "torch.DoubleTensor")
      tester:eq(torch.typename(module.forwardnodes[1].children[1].data.input[1]), "torch.DoubleTensor")
      tester:eq(torch.typename(module.backwardnodes[1].children[1].data.gradOutput[1]), "torch.DoubleTensor")

      module:float()
      tester:eq(torch.typename(module.output), "torch.FloatTensor")
      tester:eq(torch.typename(module.gradInput), "torch.FloatTensor")
      tester:eq(torch.typename(module.innode.data.input[1]), "torch.FloatTensor")
      tester:eq(torch.typename(module.outnode.data.input[1]), "torch.FloatTensor")
      tester:eq(torch.typename(module.forwardnodes[1].data.input[1]), "torch.FloatTensor")
      tester:eq(torch.typename(module.forwardnodes[1].children[1].data.input[1]), "torch.FloatTensor")
      tester:eq(torch.typename(module.backwardnodes[1].children[1].data.gradOutput[1]), "torch.FloatTensor")
      local output = module:forward(input:float())
      tester:eq(torch.typename(output), "torch.FloatTensor")
      local gradInput = module:backward(input:float(), gradOutput:float())
      tester:eq(torch.typename(gradInput), "torch.FloatTensor")

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

      tester:assert(hidden_a:label():match('DescB') ~= nil)
      local fg_tmpfile = os.tmpname()
      local bg_tmpfile = os.tmpname()
      if not pcall(function() graph.dot(net.fg, 'Test', fg_tmpfile) end) then
         return -- prevent graphviz not found error
      end
      graph.dot(net.fg, 'Test BG', bg_tmpfile)

      local function checkDotFile(tmpfile)
         local dotcontent = io.open(tmpfile .. '.dot', 'r'):read("*all")
         tester:assert(
         dotcontent:match('%[color=red.*label=%"Input.*DescA.*%".*%]') ~= nil)
         tester:assert(
         dotcontent:match(
         '%[.*fontcolor=green.*label=%"Hidden A.*DescB.*%".*%]') ~= nil)
         tester:assert(
         dotcontent:match('%[color=blue.*label=%".*DescB.*%".*%]') ~= nil)
         tester:assert(
         dotcontent:match(
         '%[.*label=%".*DescB.*%".*tooltip=%"I am green%".*%]') ~= nil)
      end

      checkDotFile(fg_tmpfile)
      checkDotFile(bg_tmpfile)
   end

   function test.test_splitMore()
      local nSplits = 2
      local in1 = nn.Identity()()
      local out1, out2 = nn.SplitTable(2)(in1):split(nSplits)

      local model = nn.gModule({in1}, {out1, out2})
      local input = torch.randn(10, nSplits + 1)
      local ok, result = pcall(model.forward, model, input)
      assert(not ok, "the extra input to split should be detected")
   end

   function test.test_splitLess()
      local nSplits = 3
      local in1 = nn.Identity()()
      local out1, out2, out3 = nn.SplitTable(2)(in1):split(nSplits)

      local model = nn.gModule({in1}, {out1, out2, out3})
      local input = torch.randn(10, nSplits - 1)
      local ok, result = pcall(model.forward, model, input)
      assert(not ok, "the missing input to split should be detected")
   end

   function test.test_gradOutputZeroOptim()
      local unpack = function(...)
	 if _G[unpack] then return _G[unpack](...)
	 else return table.unpack(...) end
      end
      -- Make module that produces an expanded zero gradInput tensor
      local dummyModule = nn.Module()
      dummyModule.updateOutput = function(self, input)
         self.output = torch.Tensor(1, 2, 3):uniform()
         return self.output
      end
      dummyModule.updateGradInput = function(self, input, gradOutput)
         local zeroTensor = torch.Tensor{0}
          :view(unpack(torch.ones(input:dim()):totable()))
          :expandAs(input)
         self.gradInput = zeroTensor
         return self.gradInput
      end

      -- First input and final gradOutput
      local input = torch.Tensor(1, 2, 3):uniform()
      local gradOutput = torch.Tensor(1, 2, 3):uniform()

      -- First case: one intermediary gradOutput is going to be zero
      local x = nn.Identity()()
      local h1 = dummyModule:clone()(x)
      local h2 = nn.Identity()(x)
      local y = nn.CAddTable()({h1, h2})
      local mod = nn.gModule({x}, {y})

      local ok, result = pcall(nn.Module.forward, mod, input)
      assert(ok, "forward should succeed")

      nn.Module.backward( mod, input, gradOutput)
      ok, result = pcall(nn.Module.backward, mod, input, gradOutput)
      assert(ok, "backward should succeed")

      -- Second case: all intermediary gradOutputs are going to be zero
      local x = nn.Identity()()
      local h1 = dummyModule:clone()(x)
      local h2 = dummyModule:clone()(x)
      local y = nn.CAddTable()({h1, h2})
      local mod = nn.gModule({x}, {y})

      local ok, result = pcall(nn.Module.forward, mod, input)
      assert(ok, "forward should succeed")

      ok, result = pcall(nn.Module.backward, mod, input, gradOutput)
      assert(ok, "backward should succeed")
   end

   function test.test_replace()
      local i = nn.Identity()()
      local l1 = nn.Linear(5, 2)(i)
      local sig = nn.Sigmoid()(l1)
      local l2  = nn.Linear(2, 5)(sig)
      local model = nn.gModule({i}, {l2})

      local input = torch.randn(4, 5)
      local gradOutput = torch.randn(4, 5)
      tester:eq(model:forward(input):size(), input:size(), "inconsistent output size")
      tester:eq(model:backward(input, gradOutput):size(), input:size(), "inconsistent output size")

      model:replace(function(m)
         if torch.type(m) == 'nn.Linear' then
            if m.weight:size(1) == 5 then
               return nn.Linear(2, 10)
            elseif m.weight:size(1) == 2 then
               return nn.Linear(10, 2)
            end
         elseif torch.type(m) == 'nn.Sigmoid' then
            return nn.Tanh()
         end
         return m
      end)

      local input = torch.randn(4, 10)
      local gradOutput = torch.randn(4, 10)
      tester:eq(model:forward(input):size(), input:size(), "inconsistent output size")
      tester:eq(model:backward(input, gradOutput):size(), input:size(), "inconsistent output size")

      tester:ne(model.modules[2], l1, "gModule.modules wasn't updated")
      tester:ne(model.modules[3], sig, "gModule.modules wasn't updated")
      tester:eq(torch.type(model.modules[3]), 'nn.Tanh', "replace didn't update gModule.modules")
      tester:ne(model.modules[4], l2, "gModule.modules wasn't updated")
   end

   tester:add(test):run()
