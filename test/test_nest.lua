
require 'totem'
require 'nngraph'

local test = {}
local tester = totem.Tester()

function test.test_output()
   local in1 = nn.Identity()()
   local in2 = nn.Identity()()
   local in3 = nn.Identity()()
   local ok = nn.CAddTable()(nngraph.nest({in1}))
   local in1Again = nngraph.nest(in1)
   local state = nngraph.nest({in1, {in2}, in3})

   local net = nn.gModule(
      {in1, in2, in3},
      {ok, in1Again, state, nngraph.nest({in3}), nngraph.nest({in1, in2})})

   local val1 = torch.randn(7, 3)
   local val2 = torch.randn(2)
   local val3 = torch.randn(3)
   local expectedOutput = {
      val1, val1, {val1, {val2}, val3}, {val3}, {val1, val2},
   }
   local output = net:forward({val1, val2, val3})
   tester:eq(output, expectedOutput, "output")
end


return tester:add(test):run()


