
local nesting = {}

local utils = paths.dofile('utils.lua')
local istensor = utils.istensor


-- Creates a clone of a tensor or of a table with tensors.
function nesting.cloneNested(obj)
	if istensor(obj) then
		return obj:clone()
	end

	local result = {}
	for key, child in pairs(obj) do
		result[key] = nesting.cloneNested(child)
	end
	return result
end

-- Fills the obj with the given value.
-- The obj can be a tensor or a table with tensors.
function nesting.fillNested(obj, value)
	if istensor(obj) then
		obj:fill(value)
	else
		for key, child in pairs(obj) do
			nesting.fillNested(child, value)
		end
	end
end

-- Resizes all tensors in the output.
function nesting.resizeNestedAs(output, input)
	if istensor(output) then
		output:resizeAs(input)
	else
		for key, child in pairs(input) do
			-- A new element is added to the output, if needed.
			if not output[key] then
				output[key] = nesting.cloneNested(child)
			else
				nesting.resizeNestedAs(output[key], child)
			end
		end
		-- Extra elements are removed from the output. 
		for key, child in pairs(output) do
			if not input[key] then
				output[key] = nil
			end
		end
	end
end

-- Adds the input to the output.
-- The input can contain nested tables.
-- The output will contain the same nesting of tables.
function nesting.addNestedTo(output, input)
	if istensor(output) then
		output:add(input)
	else
		for key, child in pairs(input) do
			assert(output[key] ~= nil, "missing key")
			nesting.addNestedTo(output[key], child)
		end
	end
end

return nesting
