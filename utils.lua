local utils = {}

function utils.istorchclass(x)
   return type(x) == 'table' and torch.typename(x)
end

function utils.istable(x)
   return type(x) == 'table' and not torch.typename(x)
end

--[[ Returns a useful error message when a nngraph.Node is expected. ]]
function utils.expectingNodeErrorMessage(badVal, array, idx)
   if badVal == nil then
      return string.format('%s[%d] is nil (typo / bad index?)', array, idx)
   elseif torch.isTypeOf(badVal, 'nn.Module') then
      local errStr = '%s[%d] is an nn.Module, specifically a %s, but the ' ..
                     'only valid thing to pass is an instance of ' ..
                     'nngraph.Node. Did you forget a second set of parens, ' ..
                     'which convert a nn.Module to a nngraph.Node?'
      return string.format(errStr, array, idx, torch.typename(badVal))
   else
      local errStr = '%s[%d] should be an nngraph.Node but is of type %s'
      return string.format(errStr, array, idx,
                         torch.typename(badVal) or type(badVal))
   end
end

--[[ Lua 5.2+ removed table.maxn, provide fallback implementation. ]]
if table.maxn then
   utils.tableMaxN = table.maxn
else
   function utils.tableMaxN(tbl)
      local max = 0
      for k, v in pairs(tbl) do
         if type(k) == 'number' and k > max then
            max = k
         end
      end
      return max
   end
 end
return utils
