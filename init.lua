
require 'nn'
require 'graph'

nngraph = {}

torch.include('nngraph','node.lua')
torch.include('nngraph','gmodule.lua')
torch.include('nngraph','modules.lua')

-- handy functions
local utils = paths.dofile('utils.lua')
local istensor = utils.istensor
local istable = utils.istable
local istorchclass = utils.istorchclass



-- Modify the __call function to hack into nn.Module
local Module = torch.getmetatable('nn.Module')
function Module:__call__(input,noutput)

	if not noutput and type(input) == 'number' then
		noutput = input
		input = {}
	end
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
		mnode:add(dnode,true)
	end

	if noutput == nil then
		return mnode
	end

	-- backward compatibility for a while, reaise an error.
	error('Use node:split(noutput) to split the output of a node into multiple nodes')
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
