require 'nngraph'
function t1()
	local m0=nn.Linear(20,1)(nn.Tanh()(nn.Linear(20,20)()))
	local m1=nn.Linear(10,1)(nn.Tanh()(nn.Linear(10,10)()))
	local madd=nn.CAddTable()({m0,m1})
	local m2=nn.Sigmoid()(madd)
	local m3=nn.Tanh()(madd)
	local x = torch.rand(20)
	local y = torch.rand(10)
	local gmod = nn.gModule({m2,m3})
	gmod.verbose = true
	print('forward')
	gmod:updateOutput({x,y})
	print('updateGradInput')
	gmod:updateGradInput({x,y},{torch.rand(1),torch.rand(1)})
	graph.dot(gmod.fg)
	graph.dot(gmod.bg)
end

function t2()
	print('compare')
	local m0 = nn.Linear(5,10)()
	local m1 = nn.Linear(10,20)()
	local m2 = nn.Linear(30,50)(nn.JoinTable(1){m0,m1})
	local gmod = nn.gModule(m2)

	local nn0 = nn.Linear(5,10)
	local nn1 = nn.Linear(10,20)
	local nn2 = nn.Linear(30,50)
	local nnmod = nn.Sequential():add(nn.ParallelTable():add(nn0):add(nn1)):add(nn.JoinTable(1)):add(nn2)

	nn0.weight:copy(m0.data.module.weight)
	nn0.bias:copy(m0.data.module.bias)
	nn1.weight:copy(m1.data.module.weight)
	nn1.bias:copy(m1.data.module.bias)
	nn2.weight:copy(m2.data.module.weight)
	nn2.bias:copy(m2.data.module.bias)


	for i=1,5 do
		local x,y = torch.rand(5),torch.rand(10)
		local xx,yy = x:clone(),y:clone()

		gmod:updateOutput({x,y})
		nnmod:updateOutput({xx,yy})
		print('fdiff = ', torch.dist(gmod.output,nnmod.output))

		local odx = torch.rand(50)
		local odxx = odx:clone()

		gmod:updateGradInput({x,y},odx)
		nnmod:updateGradInput({xx,yy},odxx)
		graph.dot(gmod.fg)
		for i,v in ipairs(gmod.gradInput) do
			print('bdiff [' ..i..  '] = ', torch.dist(gmod.gradInput[i],nnmod.gradInput[i]))
		end
	end

	local gms = {m0,m1,m2}
	local nms = {nn0,nn1,nn2}

	for i=1,5 do
		local x,y = torch.rand(5),torch.rand(10)
		local xx,yy = x:clone(),y:clone()

		gmod:updateOutput({x,y})
		nnmod:updateOutput({xx,yy})
		print('fdiff = ', torch.dist(gmod.output,nnmod.output))

		local odx = torch.rand(50)
		local odxx = odx:clone()

		gmod:zeroGradParameters()
		nnmod:zeroGradParameters()

		gmod:updateGradInput({x,y},odx)
		nnmod:updateGradInput({xx,yy},odxx)

		gmod:accGradParameters({x,y},odx)
		nnmod:accGradParameters({xx,yy},odxx)
		graph.dot(gmod.fg)
		for i,v in ipairs(gms) do
			print('accdiff [' ..i..  '] = ', torch.dist(gms[i].data.module.gradWeight,nms[i].gradWeight))
			print('accdiff [' ..i..  '] = ', torch.dist(gms[i].data.module.gradBias,nms[i].gradBias))
		end
	end
end

t2()
