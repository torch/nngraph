
require 'totem'
require 'nngraph'
local test = {}
local tester = totem.Tester()

function test.test_output()
   local input = torch.randn(7, 11)
   local module = nngraph.JustTable()
   tester:eq(module:forward(input), {input}, "output")
end

function test.test_grad()
   local input = torch.randn(7, 11)
   local module = nngraph.JustTable()
   totem.nn.checkGradients(tester, module, input)
end

tester:add(test):run()
