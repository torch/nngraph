
require 'totem'
require 'nngraph'
local test = {}
local tester = totem.Tester()

function test.test_call()
   local prediction = nn.Identity()()
   local target = nn.Identity()()
   local mse = nn.MSECriterion()({prediction, target})
   local costBits = nn.MulConstant(1/math.log(2))(mse)
   local net = nn.gModule({prediction, target}, {costBits})

   local input = {torch.randn(3, 5), torch.rand(3, 5)}
   local criterion = nn.MSECriterion()
   local output = net:forward(input)
   criterion:forward(input[1], input[2])
   tester:eq(output[1], criterion.output/math.log(2), "output", 1e-14)

   local gradOutput = torch.randn(1)
   local gradInput = net:backward(input, gradOutput)
   criterion:backward(input[1], input[2])
   tester:eq(gradInput[1], criterion.gradInput:clone():mul(gradOutput[1]/math.log(2)), "gradPrediction", 1e-14)
   tester:eq(gradInput[2], torch.zeros(input[2]:size()), "gradTarget")
end

function test.test_grad()
   local prediction = nn.Identity()()
   local zero = nn.MulConstant(0)(prediction)
   -- The target is created inside of the nngraph
   -- to ignore the zero gradTarget.
   local target = nn.AddConstant(1.23)(zero)
   local mse = nn.MSECriterion()({prediction, target})
   local net = nn.gModule({prediction}, {mse})

   local input = torch.randn(4, 7)
   totem.nn.checkGradients(tester, net, input)
end

local function module()
   local module = nn.ModuleFromCriterion(nn.MSECriterion())
   local input = {torch.randn(3, 5), torch.randn(3, 5)}
   return module, input
end

function test.test_serializable()
   local module, input = module()
   totem.nn.checkSerializable(tester, module, input)
end

function test.test_typeCastable()
   local module, input = module()
   totem.nn.checkTypeCastable(tester, module, input)
end


tester:add(test):run()
