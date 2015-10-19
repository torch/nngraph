
local JustTable, parent = torch.class('nngraph.JustTable', 'nn.Module')
function JustTable:__init()
   self.output = {}
end

-- The input is one element.
-- The output is a table with one element: {element}
function JustTable:updateOutput(input)
   self.output[1] = input
   return self.output
end

function JustTable:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput[1]
   return self.gradInput
end
