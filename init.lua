
require 'nn'
require 'graph'

nngraph = {}

torch.include('nngraph','node.lua')
torch.include('nngraph','gmodule.lua')

local function istensor(x)
	if torch.typename(x) and torch.typename(x):find('Tensor') then
		return true
	end
	return false
end

local function istorchclass(x)
	return type(x) == 'table' and torch.typename(x)
end

local function istable(x)
	return type(x) == 'table' and not torch.typename(x)
end

-- Modify the __call function to hack into nn.Module
local Module = torch.getmetatable('nn.Module')
function Module:__call__(input)

	if not istable(input) then
		input = {input}
	end
	local mnode = nngraph.Node({module=self})
	local dnodes = {}
	for i,dnode in ipairs(input) do
		if torch.typename(dnode) ~= 'nngraph.Node' then
			if istensor(dnode) then
				dnode = nngraph.Node({input=dnode})
			elseif istorchclass(dnode) then
				dnode = nngraph.Node({module=dnode})
			else
				error('what is this in the input? ' .. dnode)
			end
		end
		mnode:add(dnode)
	end
	return mnode
end

-- Modify the __call function to hack into nn.Criterion
local Criterion = torch.getmetatable('nn.Criterion')
function Criterion:__call__(input)
	if not istable(input) then
		input = {input}
	end
	local mnode = nngraph.Node({criterion=self})
	local dnodes = {}
	for i,dnode in ipairs(input) do
		if torch.typename(dnode) ~= 'nngraph.Node' then
			if istensor(dnode) then
				dnode = nngraph.Node({input=dnode})
			elseif istorchclass(dnode) then
				dnode = nngraph.Node({module=dnode})
			else
				error('what is this in the input? ' .. dnode)
			end
		end
		mnode:add(dnode)
	end
	return mnode
end
