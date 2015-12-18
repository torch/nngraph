
require 'totem'
require 'nngraph'
local test = {}
local tester = totem.Tester()

function test.test_output()
   local input = {torch.randn(7, 11)}
   local module = nngraph.JustElement()
   tester:eq(module:forward(input), input[1], "output")
end

function test.test_grad()
   local input = {torch.randn(7, 11)}
   local module = nngraph.JustElement()
   totem.nn.checkGradients(tester, module, input)
end

function test.test_split()
   local in1 = nn.Identity()()
   local output = in1:split(1)
   local net = nn.gModule({in1}, {output})

   local input = {torch.randn(7, 11)}
   tester:eq(net:forward(input), input[1], "output of split(1)")
end

tester:add(test):run()
