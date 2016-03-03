
local nesting = {}

local utils = require('nngraph.utils')

-- Creates a clone of a tensor or of a table with tensors.
function nesting.cloneNested(obj)
   if torch.isTensor(obj) then
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
   if torch.isTensor(obj) then
      obj:fill(value)
   else
      for key, child in pairs(obj) do
         nesting.fillNested(child, value)
      end
   end
end

-- Resizes all tensors in the output.
function nesting.resizeNestedAs(output, input)
   if torch.isTensor(output) then
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

-- Copies all tensors in the output.
function nesting.copyNested(output, input)
   if torch.isTensor(output) then
      output:copy(input)
   else
      for key, child in pairs(input) do
          nesting.copyNested(output[key], child)
      end
      -- Extra elements in the output table cause an error.
      for key, child in pairs(output) do
         if not input[key] then
            error('key ' .. tostring(key) ..
                  ' present in output but not in input')
         end
      end
   end
end

-- Adds the input to the output.
-- The input can contain nested tables.
-- The output will contain the same nesting of tables.
function nesting.addNestedTo(output, input)
   if torch.isTensor(output) then
      output:add(input)
   else
      for key, child in pairs(input) do
         assert(output[key] ~= nil, "missing key")
         nesting.addNestedTo(output[key], child)
      end
   end
end

return nesting
