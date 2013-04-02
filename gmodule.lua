
local function istable(x)
	return type(x) == 'table' and not torch.typename(x)
end

local gModule, parent = torch.class('nn.gModule','nn.Module')

function gModule:__init(...)
	parent.__init(self)
	local nodes = {...}
	-- the graph is defined backwards, we have the output modules as input here
	-- we will define a dummy output node that connects all output modules
	-- into itself. This will be the output for the forward graph and
	-- input point for the backward graph
	local outnode = nngraph.Node({input={}})
	for i,n in ipairs(nodes) do
		outnode:add(n)
	end

	-- the backward graph (bg) is for gradients
	-- the forward graph (fg) is for function evaluation
	self.bg = outnode:graph()
	self.fg = outnode:graph():reverse()

	-- these are the input modules (there can be more than one)
	self.roots = self.fg:roots()

	-- now we do the same thing as we did for outnode, we create dummy input node
	-- and connect all input modules to this one.
	local innode = nngraph.Node({data={}})
	for i,root in ipairs(self.roots) do
		innode:add(root)
	end

	-- the complete graph is constructed
	-- now regenerate the graphs with the additional nodes
	self.fg = innode:graph()
	self.bg = self.fg:reverse()

	self.innode = innode
	self.outnode = self.bg:roots()[1]
	self.verbose = false

	if #nodes > 1 then
		self.output = {}
		for i,node in ipairs(nodes) do
			table.insert(self.output,node.data.module and node.data.module.output or node.data.input)
		end
	else
		local node = nodes[1]
		self.output = node.data.module and node.data.module.output or node.data.input
	end

	if #self.roots > 1 then
		self.gradInput = {}
		for i,node in ipairs(self.roots) do
			table.insert(self.gradInput,node.data.module and node.data.module.gradInput or nil)
		end
	else
		node = self.roots[1]
		self.gradInput = node.data.module and node.data.module.gradInput or nil
	end

end

function gModule:updateOutput(input)
	-- we will assume that the input is either a table of stuff
	-- if not we will put it in a table of stuff
	if torch.typename(input) or type(input) ~= 'table' then
		input={input}
	end
	local function neteval(node)
		if node.data.data then
			-- then this is a data node, just propagate into
			-- its children
			-- this is different from a regular data node
			-- the input is expected to be a table of things
			-- where each thing goes into the input of 
			-- corresponding children. So this is like a
			-- dispatcher
			for i,child in ipairs(node.children) do
				if child.data.input then
					table.insert(child.data.input,node.data.data[i])
				else
					child.data.input = {node.data.data[i]}
				end
			end
		elseif not node.data.module and not node.data.criterion and node.data.input then
			-- then this is a data node, just propagate into
			-- its children
			for i,child in ipairs(node.children) do
				if child.data.input then
					table.insert(child.data.input,#node.data.input == 1 and node.data.input[1] or node.data.input)
				else
					child.data.input = {#node.data.input == 1 and node.data.input[1] or node.data.input}
				end
			end
		elseif node.data.module then
			local module = node.data.module
			local input = node.data.input
			if #input == 1 then
				input = input[1]
			end
			-- forward through this node
			local output = module:updateOutput(input)
			-- propagate the output to children
			for i,child in ipairs(node.children) do
				if child.data.input then
					table.insert(child.data.input,output)
				else
					child.data.input = {output}
				end
			end
		elseif node.data.criterion then
			local module = node.data.criterion
			local input = node.data.input
			-- forward through this node
			local output = module:updateOutput(unpack(input))
			-- propagate the output to children
			for i,child in ipairs(node.children) do
				if child.data.input then
					table.insert(child.data.input,output)
				else
					child.data.input = {output}
				end
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

	-- set the data field to current input
	local innode = self.innode
	innode.data.data=input
	if #self.roots ~= #innode.children then
		print('#inputs =' .. #innode.children)
		print('#roots  =' .. #self.roots)
		error('Number of inputs do not match my graph')
	end
	-- first clear the input states
	innode:bfs(function(node) node.data.input = nil end)
	-- the run forward
	innode:bfs(neteval)

	-- everything is done, so now I can collect the results
	-- that are stored in outnode.input
	-- local outputs = self.outnode.data.input
	-- self.output = #outputs == 1 and outputs[1] or outputs
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
		if node.data.data then
			-- then this is a data node, just propagate into
			-- its children
			-- this is different from a regular data node
			-- the input is expected to be a table of things
			-- where each thing goes into the input of 
			-- corresponding children. So this is like a
			-- dispatcher
			for i,child in ipairs(node.children) do
				if child.data.gradOutput then
					table.insert(child.data.gradOutput,node.data.data[i])
				else
					child.data.gradOutput = {node.data.data[i]}
				end
			end
		elseif not node.data.module and node.data.gradOutput then
			-- then this is a data node, just propagate into
			-- its children
			for i,child in ipairs(node.children) do
				if child.data.gradOutput then
					table.insert(child.data.gradOutput,node.data.gradOutput)
				else
					child.data.gradOutput = {node.data.gradOutput}
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
				for i=2,#gradOutput do
					gradOutput[1]:add(gradOutput[i])
				end
				gradOutput = gradOutput[1]
			end
			local gradInput = module:updateGradInput(input,gradOutput)
			-- propagate the output to children
			for i,child in ipairs(node.children) do
				local gi
				if istable(gradInput) and istable(input) then
					gi = gradInput[i]
				else
					gi = gradInput
				end
				if child.data.gradOutput then
					table.insert(child.data.gradOutput,gi)
				else
					child.data.gradOutput = {gi}
				end
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
	outnode:bfs(function(node) node.data.gradOutput = nil end)
	outnode:bfs(neteval)

	-- self.gradInput = self.innode.data.gradOutput
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
			-- then this is a data node, just propagate into
			-- its children
			-- this is different from a regular data node
			-- the input is expected to be a table of things
			-- where each thing goes into the input of 
			-- corresponding children. So this is like a
			-- dispatcher
			for i,child in ipairs(node.children) do
				if child.data.gradOutput then
					table.insert(child.data.gradOutput,node.data.data[i])
				else
					child.data.gradOutput = {node.data.data[i]}
				end
			end
		elseif not node.data.module and node.data.gradOutput then
			-- then this is a data node, just propagate into
			-- its children
			for i,child in ipairs(node.children) do
				if child.data.gradOutput then
					table.insert(child.data.gradOutput,node.data.gradOutput)
				else
					child.data.gradOutput = {node.data.gradOutput}
				end
			end
		elseif node.data.module then
			local module = node.data.module
			local gradOutput = node.data.gradOutput
			local input = node.data.input
			if #input == 1 then
				input = input[1]
			end
			-- accGradParameters through this node
			if istable(gradOutput) and not istable(module.output) then
				for i=2,#gradOutput do
					gradOutput[1]:add(gradOutput[i])
				end
				gradOutput = gradOutput[1]
			end
			module:accGradParameters(input,gradOutput,lr)
			local gradInput = module.gradInput
			-- propagate the output to children
			for i,child in ipairs(node.children) do
				local gi
				if istable(gradInput) and istable(input) then
					gi = gradInput[i]
				else
					gi = gradInput
				end
				if child.data.gradOutput then
					table.insert(child.data.gradOutput,gi)
				else
					child.data.gradOutput = {gi}
				end
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
	outnode:bfs(function(node) node.data.gradOutput = nil end)
	outnode:bfs(neteval)
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
