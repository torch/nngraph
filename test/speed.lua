
require 'nngraph'

function time_benchmark(model, input, n)
   local forward_timer = torch.Timer():stop():reset()
   local backward_timer = torch.Timer():stop():reset()
   local total_timer = torch.Timer():stop():reset()
   local gradOut
   total_timer:resume()
   for i = 1, n do
      forward_timer:resume()
      local out = model:forward(input)
      forward_timer:stop()
      if not gradOut then
         gradOut = torch.rand(out:size())
      end
      backward_timer:resume()
      model:backward(input, gradOut)
      backward_timer:stop()
   end
   total_timer:stop()

   return {forward = forward_timer:time().real,
   backward = backward_timer:time().real,
   total = total_timer:time().real}
end

function report_benchmark(result, title)
   local nspace = (80-string.len(title))/2
   report = {string.rep('#', 80),
   string.format('%s%s%s', string.rep(' ', math.floor(nspace)), title, string.rep(' ', math.ceil(nspace))),
   string.format('Total Time Spent = %.2f s', result.total),
   string.format('    Forward Time = %.2f s', result.forward),
   string.format('   Backward Time = %.2f s', result.backward),
   string.rep('#', 80)
}
print(table.concat(report, '\n'))
return result
end

function compare_benchmarks(result, base, title)
   local nspace = (80-string.len(title))/2
   report = {string.rep('#', 80),
   string.format('%s%s%s', string.rep(' ', math.floor(nspace)), title, string.rep(' ', math.ceil(nspace))),
   string.format('Total Time Spent = %.2f %%', result.total/base.total*100),
   string.format('    Forward Time = %.2f %%', result.forward/base.forward*100),
   string.format('   Backward Time = %.2f %%', result.backward/base.backward*100),
   string.rep('#', 80)
}
print(table.concat(report, '\n'))
return result
end

function get_models(nhidden_layers, ninput, noutput, nhidden)

   local function get_concat_layer(nfrom, nto)
      local concat_module = nn.Sequential()
      local concat_layer = nn.ConcatTable()
      concat_layer:add(nn.Linear(nfrom, nto))
      concat_layer:add(nn.Linear(nfrom, nto))
      concat_module:add(concat_layer)
      concat_module:add(nn.CAddTable())
      concat_module:add(nn.ReLU())
      return concat_module
   end

   -- NN
   local nn_model = nn.Sequential()
   nn_model:add(get_concat_layer(ninput, nhidden))--nn.Linear(ninput, nhidden)):add(nn.ReLU())
   for i = 1, nhidden_layers do
      nn_model:add(get_concat_layer(nhidden, nhidden))--nn.Linear(nhidden, nhidden)):add(nn.ReLU())
   end
   nn_model:add(get_concat_layer(nhidden, noutput))--nn.Linear(nhidden, noutput))

   -- NN graph
   local input = nn.Identity()()
   local nng_model = nn.ReLU()(nn.CAddTable()({nn.Linear(ninput, nhidden)(input),
   nn.Linear(ninput, nhidden)(input)}))
   for i = 1, nhidden_layers do
      nng_model = nn.ReLU()(nn.CAddTable()({nn.Linear(nhidden, nhidden)(nng_model),
      nn.Linear(nhidden, nhidden)(nng_model)}))
   end
   nng_model = nn.ReLU()(nn.CAddTable()({nn.Linear(nhidden, noutput)(nng_model),
   nn.Linear(nhidden, noutput)(nng_model)}))

   nng_model = nn.gModule({input},{nng_model})

   return {nn_model = nn_model, nng_model = nng_model}
end

function get_options(arg)
   local cmd = torch.CmdLine()
   cmd:text('nngraph benchmarking')
   cmd:option('-niter', 10, 'number of iterations of forward/backward for each model')
   cmd:option('-nhidden_layers', 10, 'number of hidden layers')
   cmd:option('-input_size', 512, 'size of input')
   cmd:option('-batch_size', 16, 'size of batch')
   cmd:option('-hidden_size', 1024, 'size of hidden layer')
   cmd:option('-output_size', 128, 'size of output layer')
   local opt = cmd:parse(arg)
   return opt
end

local opt = get_options(arg)
models = get_models(opt.nhidden_layers, opt.input_size, opt.output_size, opt.hidden_size)
print(opt)

local nn_bench = report_benchmark(time_benchmark(models.nn_model, torch.rand(opt.batch_size,opt.input_size), opt.niter), 'NN')
local nng_bench = report_benchmark(time_benchmark(models.nng_model, torch.rand(opt.batch_size,opt.input_size), opt.niter), 'NNGRAPH')
compare_benchmarks(nng_bench, nn_bench, 'NNGRAPH / NN (%)')

