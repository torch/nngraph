
local JustElement, parent = torch.class('nngraph.JustElement', 'nn.Module')
function JustElement:__init()
   self.gradInput = {}
end

-- The input is a table with one element.
-- The output the element from the table.
function JustElement:updateOutput(input)
   assert(#input == 1, "expecting one element")
   self.output = input[1]
   return self.output
end

function JustElement:updateGradInput(input, gradOutput)
   self.gradInput[1] = gradOutput
   return self.gradInput
end
