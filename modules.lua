
local identitytable = torch.class('nn.IdentityTable','nn.Module')

function identitytable:__init()
	self.output={}
	self.gradInput={}
end

function identitytable:updateOutput(input)
	while #self.output>0 do table.remove(self.output) end
	for i,inp in ipairs(input) do table.insert(self.output,inp) end
	return self.output
end

function identitytable:updateGradInput(input,gradOutput)
	while #self.gradInput>0 do table.remove(self.gradInput) end
	for i,inp in ipairs(gradOutput) do table.insert(self.gradInput,inp) end
	return self.gradInput
end
