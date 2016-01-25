local totem = require 'totem'
require 'nngraph'
local tests = totem.TestSuite()
local tester = totem.Tester()

function tests.connectivity()
  -- Store debug info here, need to call debug.getinfo on same line as the
  -- dangling pointer is declared.
  local dInfo
  local input = nn.Identity()()
  local lin = nn.Linear(20, 10)(input)
  -- The Sigmoid does not connect to the output, so should cause an error
  -- when we call gModule.
  local dangling = nn.Sigmoid()(lin); dInfo = debug.getinfo(1, 'Sl')
  local actualOutput = nn.Tanh()(lin)
  local errStr = string.format(
      'node declared on %%[%s%%]:%d_ does not connect to gmodule output',
      dInfo.short_src, dInfo.currentline)
  tester:assertErrorPattern(
      function()
        return nn.gModule({input}, {actualOutput})
      end,
      errStr)
end

return tester:add(tests):run()
