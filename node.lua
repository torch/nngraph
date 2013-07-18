
local utils = paths.dofile('utils.lua')
local istensor = utils.istensor
local istable = utils.istable
local istorchclass = utils.istorchclass


local nnNode,parent = torch.class('nngraph.Node','graph.Node')

function nnNode:__init(data)
	parent.__init(self,data)
	self.data.mapindex = self.data.mapindex or {}
end

-- domap ensures that this node will keep track of the order its children are added.
-- mapindex is a forward/backward list
-- index = self.data.mapindex[child.data]
-- child.data = self.data.mapindex[index]
function nnNode:add(child,domap)
	parent.add(self,child)
	if domap then
		local mapindex = self.data.mapindex
		local data = child.data
		assert(not mapindex[data], "Don't pass the same input twice.")
		table.insert(mapindex,data)
		mapindex[data] = #mapindex
	end
end

-- this function returns noutput number of new nodes
-- that each take a single component of the output of this 
-- node in the order they are returned.
function nnNode:split(noutput)
	assert(noutput >= 2, "splitting to one output is not supported")
	local mnode = self
	local selectnodes = {}
	for i=1,noutput do
		local node = nngraph.Node({selectindex=i,input={}})
		node:add(mnode,true)
		table.insert(selectnodes,node)
	end
	return unpack(selectnodes)
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
				local obj = v.module or v.input or v.data
				local str = tostring(obj)
				if obj.modules then
					str = torch.typename(obj)
				end
				table.insert(tstr, str)
			end
			return '{' .. table.concat(tstr,',') .. '}'
		else
			return tostring(data):gsub('\n','\\l')
		end
	end

	for k,v in pairs(self.data) do
		local vstr = ''
		if k=='mapindex' then
			vstr = getmapindexstr(v)
		else
			vstr = getstr(v)
		end
		table.insert(lbl, k .. ' = ' .. vstr)
	end
	return table.concat(lbl,"\\l")
end
