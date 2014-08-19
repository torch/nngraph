
require 'nn'
require 'graph'

nngraph = {}

torch.include('nngraph','node.lua')
torch.include('nngraph','gmodule.lua')
torch.include('nngraph','graphinspecting.lua')
torch.include('nngraph','ModuleFromCriterion.lua')

-- handy functions
local utils = paths.dofile('utils.lua')
local istensor = utils.istensor
local istable = utils.istable
local istorchclass = utils.istorchclass

-- simpler todot functions
nngraph.simple_print =  paths.dofile('simple_print.lua')


-- Modify the __call function to hack into nn.Module
local Module = torch.getmetatable('nn.Module')
function Module:__call__(...)
	local nArgs = select("#", ...)
	assert(nArgs <= 1, 'Use {input1, input2} to pass multiple inputs.')

	local input = ...
	if nArgs == 1 and input == nil then
		error('what is this in the input? nil')
	end
	if not istable(input) then
		input = {input}
	end
	local mnode = nngraph.Node({module=self})

	for i,dnode in ipairs(input) do
		if torch.typename(dnode) ~= 'nngraph.Node' then
			error('what is this in the input? ' .. tostring(dnode))
		end
		mnode:add(dnode,true)
	end

	return mnode
end

local Criterion = torch.getmetatable('nn.Criterion')
function Criterion:__call__(...)
	return nn.ModuleFromCriterion(self)(...)
end
