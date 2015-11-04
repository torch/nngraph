
--[[ A wrapper to turn a criterion into a module.

The gradient with respect to the target will be zero.
--]]
local ModuleFromCriterion, parent = torch.class('nn.ModuleFromCriterion','nn.Module')
function ModuleFromCriterion:__init(criterion)
   self.criterion = criterion
   self.output = torch.Tensor(1)
   self.gradInput = {torch.Tensor(), torch.Tensor()}
end

local unpack = unpack or table.unpack -- lua52 compat

--[[ The input is a {prediction, target} pair.
The output is a tensor with one number: the criterion output.
--]]
function ModuleFromCriterion:updateOutput(input)
   local prediction, target = unpack(input)
   self.output[1] = self.criterion:updateOutput(prediction, target)
   return self.output
end

function ModuleFromCriterion:updateGradInput(input, gradOutput)
   local prediction, target = unpack(input)
   local gradPrediction = self.criterion:updateGradInput(prediction, target)
   self.gradInput[1]:resizeAs(gradPrediction):copy(gradPrediction):mul(gradOutput[1])
   self.gradInput[2]:resizeAs(target):zero()
   return self.gradInput
end
