local totem = require 'totem'
require 'nngraph'
local tests = totem.TestSuite()
local tester = totem.Tester()

function tests.whatIsThisInTheInput()
  tester:assertErrorPattern(
      function()
        local inp1, inp2 = nn.Identity()(), nn.Identity() -- missing 2nd parens
        local lin = nn.Linear(20, 10)(nn.CMulTable(){inp1, inp2})
      end,
      'inputs%[2%] is an nn%.Module, specifically a nn%.Identity, but the ' ..
      'only valid thing to pass is an instance of nngraph%.Node')

  tester:assertErrorPattern(
      function()
        -- pass-through module, again with same mistake
        local graphNode, nnModule = nn.Identity()(), nn.Identity()
        return nn.gModule({graphNode, nnModule}, {graphNode})
      end,
      'inputs%[2%] is an nn%.Module, specifically a nn%.Identity, but the ' ..
      'only valid thing to pass is an instance of nngraph%.Node')

  tester:assertErrorPattern(
      function()
        local input = nn.Identity()()
        local out1 = nn.Linear(20, 10)(input)
        local out2 = nn.Sigmoid()(input)
        local unconnectedOut = nn.Linear(2, 3)
        return nn.gModule({input}, {out1, out2, unconnectedOut})
      end,
      'outputs%[3%] is an nn%.Module, specifically a nn%.Linear, but the ' ..
      'only valid thing to pass is an instance of nngraph%.Node')

  -- Check for detecting a nil in the middle of a table.
  tester:assertErrorPattern(
      function()
        local input = nn.Identity()()
        local out1 = nn.Tanh()(input)
        local out2 = nn.Sigmoid()(input)
        -- nil here is simulating a mis-spelt variable name
        return nn.gModule({input}, {out1, nil, out2})
      end,
      'outputs%[2%] is nil %(typo / bad index%?%)')

  tester:assertErrorPattern(
      function()
        -- Typo variable name returns nil, meaning an empty table
        local input = nn.Identity()({aNonExistentVariable})
      end,
      'cannot pass an empty table of inputs%. To indicate no incoming ' ..
      'connections, leave the second set of parens blank%.')
end

function tests.splitUnused()
  -- Need to do debuginfo on the same lines as the other code here to match
  -- what debug.getinfo inside those calls will return
  local dInfoDeclare, dInfoSplit
  local input = nn.Identity()(); dInfoDeclare = debug.getinfo(1, 'Sl')
  local output, unused = input:split(2); dInfoSplit = debug.getinfo(1, 'Sl')

  local function willCrash()
    return nn.gModule({input}, {output})
  end

  -- Work out what strings will be in the error message
  local declareLoc = string.format('%%[%s%%]:%d_',
                                   dInfoDeclare.short_src,
                                   dInfoDeclare.currentline)
  local splitLoc = string.format('%%[%s%%]:%d',
                                 dInfoSplit.short_src,
                                 dInfoSplit.currentline)

  tester:assertErrorPattern(
      willCrash,
      '1 of split%(2%) outputs from the node declared at ' ..
      declareLoc .. ' split at ' .. splitLoc .. '%-mnode are unused')
end

tester:add(tests):run()
