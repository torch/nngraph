
local utils = require('nngraph.utils')
local istensor = torch.isTensor
local istable = utils.istable
local istorchclass = utils.istorchclass
require 'debug'

local nnNode,parent = torch.class('nngraph.Node','graph.Node')

function nnNode:__init(data)
   parent.__init(self,data)
   self.data.annotations = self.data.annotations or {}
   self.data.mapindex = self.data.mapindex or {}
   self.data.reverseMap = self.data.reverseMap or {}
   if not self.data.annotations._debugLabel then
      self:_makeDebugLabel(debug.getinfo(6, 'Sl'))
   end
end

--[[ Build a string label which will be used a tooltip when
making a graph.]]
function nnNode:_makeDebugLabel(dinfo)
   if dinfo then
      self.data.annotations._debugLabel = string.format('[%s]:%d_%s',
                                                        dinfo.short_src,
                                                        dinfo.currentline,
                                                        dinfo.name or '')
   end
end

-- domap ensures that this node will keep track of the order its children are added.
-- mapindex is a forward/backward list
-- index = self.data.mapindex[child.data]
-- child.data = self.data.mapindex[index]
function nnNode:add(child,domap)
   parent.add(self,child)
   if domap then
      local mapindex = self.data.mapindex
      local data = child.data
      assert(not mapindex[data], "Don't pass the same input twice.")
      table.insert(mapindex,data)
      mapindex[data] = #mapindex

      -- The "child" that is added here actually represents the input node,
      -- so we write into that node to indicate that we are downstream of it.
      -- This enables dangling pointer detection.
      local revMap = child.data.reverseMap
      assert(not revMap[self], 'this connection has already been made!')
      revMap[self] = true
   end
end

-- this function returns noutput number of new nodes
-- that each take a single component of the output of this
-- node in the order they are returned.
function nnNode:split(noutput)
   if noutput == 1 then
     return nngraph.JustElement()(self)
   end
   local debugLabel = self.data.annotations._debugLabel
   -- Specify the source location where :split is called.
   local dinfo = debug.getinfo(2, 'Sl')
   local splitLoc = string.format(' split at [%s]:%d',
                                  dinfo.short_src,
                                  dinfo.currentline)
   local mnode = nngraph.Node({nSplitOutputs=noutput, annotations={_debugLabel=debugLabel .. splitLoc .. '-mnode'}})
   mnode:add(self,true)

   local selectnodes = {}
   for i=1,noutput do
      local node = nngraph.Node({selectindex=i,input={}, annotations={_debugLabel=debugLabel .. '-' .. i}})
      node:add(mnode,true)
      table.insert(selectnodes,node)
   end

   local unpack = unpack or table.unpack -- Lua52 compat
   return unpack(selectnodes)
end

function nnNode:annotate(annotations)
   for k, v in pairs(annotations) do
      self.data.annotations[k] = v
   end

   return self
end

function nnNode:graphNodeName()
   if self.data.annotations.name then
      return self.data.annotations.name .. ' (' .. self.id .. ')'
   else
      return 'Node' .. self.id
   end
end

function nnNode:graphNodeAttributes()
   self.data.annotations.graphAttributes =
   self.data.annotations.graphAttributes or {}
   if not self.data.annotations.graphAttributes.tooltip then
      self.data.annotations.graphAttributes.tooltip =
      self.data.annotations._debugLabel
   end

   return self.data.annotations.graphAttributes
end

local function getNanFlag(data)
   if data:nElement() == 0 then
      return ''
   end
   local isNan = (data:ne(data):sum() > 0)
   if isNan then
      return 'NaN'
   end
   if data:max() == math.huge then
      return 'inf'
   end
   if data:min() == -math.huge then
      return '-inf'
   end
   return ''
end

function nnNode:label()

   local lbl = {}

   local function getstr(data)
      if not data then return '' end
      if istensor(data) then
         local nanFlag = getNanFlag(data)
         local tensorType = 'Tensor'
         if data:type() ~= torch.Tensor():type() then
            tensorType = data:type()
         end
         return tensorType .. '[' .. table.concat(data:size():totable(),'x') .. ']' .. nanFlag
      elseif istable(data) then
         local tstr = {}
         for i,v in ipairs(data) do
            table.insert(tstr, getstr(v))
         end
         return '{' .. table.concat(tstr,',') .. '}'
      else
         return tostring(data):gsub('\n','\\l')
      end
   end
   local function getmapindexstr(mapindex)
      local tstr = {}
      for i,data in ipairs(mapindex) do
         local inputId = 'Node' .. (data.forwardNodeId or '')
         table.insert(tstr, inputId)
      end
      return '{' .. table.concat(tstr,',') .. '}'
   end

   for k,v in pairs(self.data) do
      local vstr = ''
      if k== 'mapindex' then
         if #v > 1 then
            vstr = getmapindexstr(v)
            table.insert(lbl, k .. ' = ' .. vstr)
         end
      elseif k== 'forwardNodeId' or k== 'annotations' then
         -- the forwardNodeId is not displayed in the label.
      else
         vstr = getstr(v)
         table.insert(lbl, k .. ' = ' .. vstr)
      end
   end

   local desc
   if self.data.annotations.description then
      desc = 'desc = ' .. self.data.annotations.description .. '\\n'
   else
      desc = ''
   end
   return desc .. table.concat(lbl,"\\l")
end
