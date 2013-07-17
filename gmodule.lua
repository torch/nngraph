
local utils = paths.dofile('utils.lua')
local istensor = utils.istensor
local istable = utils.istable
local istorchclass = utils.istorchclass

local gModule, parent = torch.class('nn.gModule','nn.Module')

function gModule:__init(inputs,outputs)
	parent.__init(self)
	-- the graph is defined backwards, we have the output modules as input here
	-- we will define a dummy output node that connects all output modules
	-- into itself. This will be the output for the forward graph and
	-- input point for the backward graph
	local outnode = nngraph.Node({input={}})
	for i,n in ipairs(outputs) do
		outnode:add(n,true)
	end
	local innode = nngraph.Node({data={},gradOutput={}})
	for i,n in ipairs(inputs) do
		n:add(innode,true)
		-- fix the mapindex for the input data node
		table.insert(innode.data.mapindex,n.data)
		innode.data.mapindex[n.data] = #innode.data.mapindex
	end

	-- the backward graph (bg) is for gradients
	-- the forward graph (fg) is for function evaluation
	self.bg = outnode:graph()
	self.fg = self.bg:reverse()

	-- the complete graph is constructed
	-- now regenerate the graphs with the additional nodes
	self.innode = self.fg:roots()[1]
	self.outnode = outnode
	self.verbose = false

	-- computation on the graph is done through topsort of forward and backward graphs
	self.forwardnodes = self.fg:topsort()
	self.backwardnodes = self.bg:topsort()

	self.output = self.outnode.data.input
	self.gradInput = self.innode.data.gradOutput

end

function gModule:apply(func)
	for i,node in ipairs(self.forwardnodes) do
		if node.data.module then
			func(node.data.module)
		end
	end
end

function gModule:updateOutput(input)
	return self:runForwardFunction('updateOutput',input)
end

function gModule:runForwardFunction(func,input)
	if type(func) == "string" then
		local func_name = func
		func = function(module,input) return module[func_name](module,input) end
	end
	-- we will assume that the input is either a table of stuff
	-- if not we will put it in a table of stuff
	if torch.typename(input) or type(input) ~= 'table' then
		input={input}
	end
	local function neteval(node)
		local function propagate(node,x)
			for i,child in ipairs(node.children) do
				child.data.input = child.data.input or {}
				local mapindex = child.data.mapindex[node.data]
				child.data.input[mapindex] = x
			end
		end
		if node.data.data then
			-- then this is a data node, just propagate into
			-- its children
			-- this is different from a regular data node
			-- the input is expected to be a table of things
			-- where each thing goes into the input of 
			-- corresponding children. So this is like a
			-- dispatcher
			-- the mapindex in a data node indexes the child data 
			-- so that this node can distribute its data to corresponding inputs
			for i,child in ipairs(node.children) do
				local mapindex = node.data.mapindex[child.data]
				if child.data.input then
					table.insert(child.data.input,node.data.data[mapindex])
				else
					child.data.input = {node.data.data[mapindex]}
				end
			end
		elseif not node.data.module and not node.data.criterion and node.data.input then
			-- then this is a data node, just propagate into
			-- its children
			local input = #node.data.input == 1 and node.data.input[1] or node.data.input
			if node.data.selectindex then
				input = input[node.data.selectindex]
			end
			propagate(node,input)
		elseif node.data.module then
			local module = node.data.module
			local input = node.data.input
			if #input == 1 then
				input = input[1]
			end
			-- forward through this node
			local output = func(module,input)
			-- propagate the output to children
			propagate(node,output)
		elseif node.data.criterion then
			local module = node.data.criterion
			local input = node.data.input
			-- forward through this node
			local output = module:updateOutput(unpack(input))
			-- propagate the output to children
			propagate(node,output)
		else
			if self.verbose then
				print('weird node, skipping :)')
				print(node.data)
			end
		end
		if self.verbose then
			print(' V : ' .. node:label())
		end
	end

	-- set the data field to current input
	local innode = self.innode
	innode.data.data=input
	if #input ~= #innode.data.mapindex then
		print('#inputs      =' .. #input)
		print('#mapindices  =' .. #innode.data.mapindex)
		error('Number of inputs do not match my graph')
	end
	-- first clear the input states
	innode:bfs(function(node)
		local input = node.data.input
		while input and #input>0 do
			table.remove(input)
		end
	end)

	-- the run forward
	for i,node in ipairs(self.forwardnodes) do
		neteval(node)
	end

	self.output = self.outnode.data.input
	if #self.outnode.children == 1 and self.output == self.outnode.data.input then
		self.output = self.output[1]
	end
	return self.output
end

function gModule:updateGradInput(input,gradOutput)
	-- we will assume that the input is either a table of stuff
	-- if not we will put it in a table of stuff
	if torch.typename(gradOutput) or type(gradOutput) ~= 'table' then
		gradOutput={gradOutput}
	end
	local outputs = {}
	local function neteval(node)
		local function propagate(node,x)
			for i,child in ipairs(node.children) do
				child.data.gradOutput = child.data.gradOutput or {}
				local mapindex = node.data.mapindex[child.data]
				table.insert(child.data.gradOutput,x[mapindex])
			end
		end
		if node.data.data then
			-- then this is a data node, just propagate into
			-- its children
			-- this is different from a regular data node
			-- the input is expected to be a table of things
			-- where each thing goes into the input of 
			-- corresponding children. So this is like a
			-- dispatcher
			-- First we need to fix the order of stuff in our
			-- gradOutput table.
			for i,child in ipairs(node.children) do
				child.data.gradOutput = child.data.gradOutput or {}
				local mapindex = node.data.mapindex[child.data]
				table.insert(child.data.gradOutput,node.data.data[mapindex])
			end
		elseif not node.data.module and node.data.gradOutput then
			-- then this is a data node, just propagate into
			-- its children
			for i,child in ipairs(node.children) do
				child.data.gradOutput = child.data.gradOutput or {}
				local go = node.data.gradOutput
				if istable(go) and #go == 1 then
					go = go[1]
				end
				if node.data.selectindex then
					child.data.gradOutput[node.data.selectindex] = go
				else
					table.insert(child.data.gradOutput,go)
				end
			end
		elseif node.data.module then
			local module = node.data.module
			local gradOutput = node.data.gradOutput
			local input = node.data.input
			if #input == 1 then
				input = input[1]
			end
			-- updateGradInput through this node
			if istable(gradOutput) and not istable(module.output) then
				if #gradOutput > 1 then
					node.data.gradOutputBuffer = node.data.gradOutputBuffer or gradOutput[1].new()
					local gobuff = node.data.gradOutputBuffer
					gobuff:resizeAs(gradOutput[1]):copy(gradOutput[1])
					for i=2,#gradOutput do
						gobuff:add(gradOutput[i])
					end
					gradOutput = gobuff
				else
					gradOutput = gradOutput[1]
				end
			elseif istable(gradOutput) and istable(module.output) and #gradOutput ~= #module.output then
				gradOutput = gradOutput[1]
			end
			local gradInput = module:updateGradInput(input,gradOutput)
			-- propagate the output to children
			for i,child in ipairs(node.children) do
				child.data.gradOutput = child.data.gradOutput or {}
				local mapindex = node.data.mapindex[child.data]
				local gi
				if #node.children ~= 1 then --istable(gradInput) and istable(input) then
					gi = gradInput[mapindex]
				else
					gi = gradInput
				end
				table.insert(child.data.gradOutput,gi)
			end
		else
			if self.verbose then
				print('weird node, skipping :)')
				print(node.data)
			end
		end
		if self.verbose then
			print(' V : ' .. node:label())
		end
	end
	local outnode = self.outnode
	outnode.data.data=gradOutput
	if #gradOutput ~= #outnode.children then
		print('#outputs   =' .. #outnode.children)
		print('#gradients =' .. #gradOutput)
		error('Number of gradients do not match my graph')
	end
	outnode:bfs(function(node)
		local gradOutput = node.data.gradOutput
		while gradOutput and #gradOutput >0 do
			table.remove(gradOutput)
		end
	end)
	for i,node in ipairs(self.backwardnodes) do
		neteval(node)
	end

	-- now fix the order of gradInput
	self.gradInput = self.innode.data.gradOutput
	if not istable(self.gradInput) then
		return self.gradInput
	end
	local gi = {}
	for i,child in ipairs(self.innode.children) do
		local mi = self.innode.data.mapindex[child.data]
		table.insert(gi,self.gradInput[mi])
	end
	while istable(self.gradInput) and #self.gradInput > 0 do
		table.remove(self.gradInput)
	end
	for i,v in ipairs(gi) do
		table.insert(self.gradInput,v)
	end

	if #self.innode.children == 1 and self.gradInput == self.innode.data.gradOutput then
		self.gradInput = self.gradInput[1]
	end

	return self.gradInput
end

function gModule:accGradParameters(input,gradOutput,lr)
	-- we will assume that the input is either a table of stuff
	-- if not we will put it in a table of stuff
	if torch.typename(gradOutput) or type(gradOutput) ~= 'table' then
		gradOutput={gradOutput}
	end
	local outputs = {}
	local function neteval(node)
		if node.data.data then
		elseif not node.data.module and node.data.gradOutput then
		elseif node.data.module then
			local module = node.data.module
			local gradOutput = node.data.gradOutput
			local input = node.data.input
			if #input == 1 then
				input = input[1]
			end
			-- accGradParameters through this node
			if istable(gradOutput) and not istable(module.output) then
				if #gradOutput > 1 then
					node.data.gradOutputBuffer = node.data.gradOutputBuffer or gradOutput[1].new()
					local gobuff = node.data.gradOutputBuffer
					gobuff:resizeAs(gradOutput[1]):copy(gradOutput[1])
					for i=2,#gradOutput do
						gobuff:add(gradOutput[i])
					end
					gradOutput = gobuff
				else
					gradOutput = gradOutput[1]
 				end
			end
			module:accGradParameters(input,gradOutput,lr)
		else
			if self.verbose then
				print('weird node, skipping :)')
				print(node.data)
			end
		end
		if self.verbose then
			print(' V : ' .. node:label())
		end
	end
	local outnode = self.outnode
	outnode.data.data=gradOutput
	if #gradOutput ~= #outnode.children then
		print('#outputs   =' .. #outnode.children)
		print('#gradients =' .. #gradOutput)
		error('Number of gradients do not match my graph')
	end
	for i,node in ipairs(self.backwardnodes) do
		neteval(node)
	end
end

function gModule:parameters()
	local p,gp = {},{}
	local innode = self.innode
	innode:bfs(function(node)
		if not node.data.module then
			return
		end

		local mp,mgp = node.data.module:parameters()
		if not mp or not mgp then return end
		for i = 1,#mp do
			table.insert(p,mp[i])
			table.insert(gp,mgp[i])
		end
	end)
	return p,gp
end
