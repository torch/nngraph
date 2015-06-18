require 'nngraph'

-- generate SVG of the graph with the problem node highlighted
-- and hover over the nodes in svg to see the filename:line_number info
-- nodes will be annotated with local variable names even if debug mode is not enabled.
nngraph.setDebug(true)

local function get_net(from, to)
    local from = from or 10
    local to = to or 10
    local input_x = nn.Identity()()
    local linear_module = nn.Linear(from, to)(input_x)

    -- Annotate nodes with local variable names
    nngraph.annotateNodes()
    return nn.gModule({input_x},{linear_module})
end

local net = get_net(10,10)

-- if you give a name to the net, it will use that name to produce the
-- svg in case of error, if not, it will come up with a name
-- that is derived from number of inputs and outputs to the graph
net.name = 'my_bad_linear_net'

-- prepare an input that is of the wrong size to force an error
local input = torch.rand(11)
pcall(function() net:updateOutput(input) end)
-- it should have produced an error and spit out a graph
-- just run Safari to display the svg
os.execute('open -a  Safari my_bad_linear_net.svg')
