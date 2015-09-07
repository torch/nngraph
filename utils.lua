local utils = {}

function utils.istorchclass(x)
   return type(x) == 'table' and torch.typename(x)
end

function utils.istable(x)
   return type(x) == 'table' and not torch.typename(x)
end

return utils
