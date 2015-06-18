-- handy functions
local utils = paths.dofile('utils.lua')
local istensor = utils.istensor
local istable = utils.istable
local istorchclass = utils.istorchclass

local function getNanFlag(data)
	if data:nElement() == 0 then
		return ''
	end
	local isNan = (data:ne(data):sum() > 0)
	if isNan then
		return 'NaN'
	end
	if data:max() == math.huge then
		return 'inf'
	end
	if data:min() == -math.huge then
		return '-inf'
	end
	return ''
end
local function getstr(data)
	if not data then return '' end
	if istensor(data) then
		local nanFlag = getNanFlag(data)
		local tensorType = 'Tensor'
		if data:type() ~= torch.Tensor():type() then
			tensorType = data:type()
		end
		return tensorType .. '[' .. table.concat(data:size():totable(),'x') .. ']' .. nanFlag
	elseif istable(data) then
		local tstr = {}
		for i,v in ipairs(data) do
			table.insert(tstr, getstr(v))
		end
		return '{' .. table.concat(tstr,',') .. '}'
	else
		return tostring(data):gsub('\n','\\l')
	end
end
local function getmapindexstr(mapindex)
	local tstr = {}
	for i,data in ipairs(mapindex) do
		local inputId = 'Node' .. (data.forwardNodeId or '')
		table.insert(tstr, inputId)
	end
	return '{' .. table.concat(tstr,',') .. '}'
end

local Node = torch.getmetatable('nngraph.Node')


--[[
Returns a textual representation of the Node that can be used by graphviz library visualization.
]]
function Node:label()

	local lbl = {}

	for k,v in pairs(self.data) do
		local vstr = ''
		if k == 'mapindex' then
			if #v > 1 then
				vstr = getmapindexstr(v)
				table.insert(lbl, k .. ' = ' .. vstr)
			end
		elseif k == 'forwardNodeId' or k == 'annotations' then
			-- the forwardNodeId is not displayed in the label.
		else
			vstr = getstr(v)
			table.insert(lbl, k .. ' = ' .. vstr)
		end
	end

	local desc = ''
	if self.data.annotations.description then
		desc = 'desc = ' .. self.data.annotations.description .. '\\n'
	end
	return desc .. table.concat(lbl,"\\l")
end

