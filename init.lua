
require 'nn'
require 'graph'

nngraph = {}

torch.include('nngraph','node.lua')
torch.include('nngraph','gmodule.lua')
torch.include('nngraph','graphinspecting.lua')

-- handy functions
local utils = paths.dofile('utils.lua')
local istensor = utils.istensor
local istable = utils.istable
local istorchclass = utils.istorchclass



-- Modify the __call function to hack into nn.Module
local Module = torch.getmetatable('nn.Module')
function Module:__call__(...)
	local nArgs = select("#", ...)
	assert(nArgs <= 2, 'Usage __call(input|{input1,input2,...} [, name])')

	local input, name = nil, nil
	if nArgs == 2 then
		input = ({...})[1]
		name = ({...})[2]
		assert(type(name) == 'string', 'The second argument can be string only (used for name)')
	elseif nArgs == 1 then
		input = ...
		if type(input) == 'string' then
			name = input
			input = nil
			nArgs = 0
		elseif input == nil then
			error('what is this in the input? nil')
		end
	end

	if not istable(input) then
		input = {input}
	end

	local mnode = nngraph.Node({module=self, _name = name})

	for i,dnode in ipairs(input) do
		if torch.typename(dnode) ~= 'nngraph.Node' then
			error('what is this in the input? ' .. tostring(dnode))
		end
		mnode:add(dnode,true)
	end

	return mnode
end
