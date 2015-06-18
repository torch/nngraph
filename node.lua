
--[[
This file implements the nngraph.Node. In addition to graph.Node this class
provides some additional functionality for handling neural networks in a graph
]]
local nnNode,parent = torch.class('nngraph.Node','graph.AnnotatedNode')


--[[
nngraph.Node
Args:
* `data` - the same as graph.Node(data). Any object type that will be stored as data 
in the graph node.
]]
function nnNode:__init(data)
	-- level 7 corresponds to level with the nngraph usage of nnNode's
	-- inside Module:__call() syntax
	parent.__init(self,data, 7)
	-- decorate the data with additional info to keep track of order of connected nodes
	self.data.mapindex = data.mapindex or {}
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
	local debugLabel = self.data.annotations._debugLabel
	local mnode = nngraph.Node({nSplitOutputs=noutput, annotations={_debugLabel=debugLabel .. '-mnode'}})
	mnode:add(self,true)

	local selectnodes = {}
	for i=1,noutput do
		local node = nngraph.Node({selectindex=i,input={}, annotations={_debugLabel=debugLabel .. '-' .. i}})
		node:add(mnode,true)
		table.insert(selectnodes,node)
	end
	return unpack(selectnodes)
end

