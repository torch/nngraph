
local function isNode(input)
   local typename = torch.typename(input)
   return typename and typename == 'nngraph.Node'
end

local function isNonEmptyList(input)
   return type(input) == "table" and #input > 0
end

local function _nest(input)
   if not isNode(input) and not isNonEmptyList(input) then
      error('what is this in the nest input? ' .. tostring(input))
   end

   if isNode(input) then
      return input
   end

   if #input == 1 then
      return nngraph.JustTable()(input)
   end

   local wrappedChildren = {}
   for i, child in ipairs(input) do
      wrappedChildren[i] = _nest(child)
   end
   return nn.Identity()(wrappedChildren)
end

-- Returns a nngraph node to represent a nested structure.
-- Usage example:
--    local in1 = nn.Identity()()
--    local in2 = nn.Identity()()
--    local in3 = nn.Identity()()
--    local ok = nn.CAddTable()(nngraph.nest({in1}))
--    local in1Again = nngraph.nest(in1)
--    local state = nngraph.nest({in1, {in2}, in3})
function nngraph.nest(...)
   local nArgs = select("#", ...)
   assert(nArgs <= 1, 'Use {input1, input2} to pass multiple inputs.')

   local input = ...
   assert(nArgs > 0 and input ~= nil, 'Pass an input.')
   return _nest(input)
end
