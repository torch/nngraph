
local function istensor(x)
	if torch.typename(x) and torch.typename(x):find('Tensor') then
		return true
	end
	return false
end

local function istable(x)
	return type(x) == 'table' and not torch.typename(x)
end

local nnNode,parent = torch.class('nngraph.Node','graph.Node')

function nnNode:__init(data)
	parent.__init(self,data)
end

function nnNode:label()

	local lbl = {}

	local function getstr(data)
		if not data then return '' end
		if istensor(data) then
			return 'Tensor[' .. table.concat(data:size():totable(),'x') .. ']'
		elseif istable(data) then
			local tstr = {}
			for i,v in ipairs(data) do
				table.insert(tstr, getstr(v))
			end
			return '{' .. table.concat(tstr,',') .. '}'
		else
			return tostring(data)
		end
	end

	for k,v in pairs(self.data) do
		table.insert(lbl, k .. ' = ' .. getstr(v))
	end
	return table.concat(lbl,"\\l")
end
