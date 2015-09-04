local function removeNodeFromEdges(node_id, edges)
   local from_nodes = {}
   local to_nodes = {}
   -- remove edges
   local idx = 1
   while idx <= #edges do
      local edge = edges[idx]
      if edge.source == node_id then
         local to_node = edges[idx].target
         table.insert(to_nodes, to_node)
         table.remove(edges, idx)
      elseif edge.target == node_id then
         local from_node = edges[idx].source
         table.insert(from_nodes, from_node)
         table.remove(edges, idx)
      else
         idx = idx + 1
      end
   end

   -- add new edges
   for _, f in pairs(from_nodes) do
      for _, t in pairs(to_nodes) do
         local edge = {source = f, target= t}
         table.insert(edges, edge)
      end
   end

   return edges
end

local function isNodeGood(node)
   return node.data and node.data.module and (torch.typename(node.data.module) ~= 'nn.Identity')
end

local function reIndexNodes(nodes, edges)
   -- make reverse map
   local rev_map = {}
   for idx = 1, #nodes do
      rev_map[nodes[idx].id] = idx
      nodes[idx].id = idx
   end
   for idx = 1, #edges do
      local edge = edges[idx]
      edge.source = rev_map[edge.source]
      edge.target = rev_map[edge.target]
   end
   return nodes, edges
end

local function cleanGraph(nodes, edges)
   local idx = 1
   while idx <= #nodes do
      local node = nodes[idx]
      if isNodeGood(node.orig_node) then
         idx = idx + 1
      else
         local id = node.id
         table.remove(nodes, idx)
         edges = removeNodeFromEdges(id, edges)
      end
   end
   return reIndexNodes(nodes, edges)
end

local function loadGraph(graph)
   local nodes = {}
   local edges = {}
   for _, node in ipairs(graph.nodes) do
      local idx = node.id
      table.insert(nodes, {id=idx, orig_node = node} )
      for ich = 1, #node.children do
         table.insert( edges, {source = idx, target = node.children[ich].id})
      end
   end
   nodes, edges = cleanGraph(nodes, edges)
   return nodes , edges
end

local M = {}

function M.todot( graph, title )
   local nodes, edges = loadGraph(graph)
   local str = {}
   table.insert(str,'digraph G {\n')
   if title then
      table.insert(str,'labelloc="t";\nlabel="' .. title .. '";\n')
   end
   table.insert(str,'node [shape = oval]; ')
   local nodelabels = {}
   for i,node in ipairs(nodes) do
      local true_node = node.orig_node
      local l =  '"' .. ( 'Node' .. true_node.id .. '\\n' .. true_node:label() ) .. '"'
      nodelabels[i] = 'n' .. true_node.id
      table.insert(str, '\n' .. nodelabels[i] .. '[label=' .. l .. '];')
   end
   table.insert(str,'\n')
   for i,edge in ipairs(edges) do
      table.insert(str,nodelabels[edge.source] .. ' -> ' .. nodelabels[edge.target] .. ';\n')
   end
   table.insert(str,'}')
   return table.concat(str,'')
end

function M.dot(g,title,fname)
   local gv = M.todot(g, title)
   local fngv = (fname or os.tmpname()) .. '.dot'
   local fgv = io.open(fngv,'w')
   fgv:write(gv)
   fgv:close()
   local fnsvg = (fname or os.tmpname()) .. '.svg'
   os.execute('dot -Tsvg -o ' .. fnsvg .. ' ' .. fngv)
   if not fname then
      require 'qtsvg'
      local qs = qt.QSvgWidget(fnsvg)
      qs:show()
      os.remove(fngv)
      os.remove(fnsvg)
      -- print(fngv,fnpng)
      return qs
   end
end

return M
