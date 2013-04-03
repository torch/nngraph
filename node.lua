
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
	self.data.mapindex = self.data.mapindex or {}
end

function nnNode:add(child,domap)
	parent.add(self,child)
	if domap then
		mapindex = self.data.mapindex
		local data = child.data
		if not mapindex[data] then
			table.insert(mapindex,data)
			mapindex[data] = #mapindex
		end
	end
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
			return tostring(data):gsub('\n','\\l')
		end
	end
	local function getmapindexstr(data)
		if not data then return '' end
		if istable(data) then
			local tstr = {}
			for i,v in ipairs(data) do
				table.insert(tstr, tostring(v.module or v.input or v.data))
			end
			return '{' .. table.concat(tstr,',') .. '}'
		else
			return tostring(data):gsub('\n','\\l')
		end
	end

	for k,v in pairs(self.data) do
		vstr = ''
		if k=='mapindex' then
			vstr = getmapindexstr(v)
		else
			vstr = getstr(v)
		end
		table.insert(lbl, k .. ' = ' .. vstr)
	end
	return table.concat(lbl,"\\l")
end
