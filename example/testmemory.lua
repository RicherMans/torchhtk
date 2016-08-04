require 'paths'
require 'torch'
require 'xlua'

local htkutils = paths.dofile('../init.lua')
require "sys"

local size = 250
local dim = 400
arr = torch.FloatTensor(size,dim)
for i=1,size do
    arr[i]:fill(i)
end

function writefile(filename)
	htkutils.write(filename,arr,"PLP")
end

local featurename = "feature.plp"

ts = sys.clock()
writefile(featurename,arr)
te = sys.clock()
print ("Time to write was " .. te - ts )
collectgarbage()

local numiters = 100000

ts= sys.clock()
-- local feat
for i=1,numiters do
  local tic = torch.tic()
  local feat = htkutils.load(featurename)
  if i%100 == 0 then
  	print(i.."/"..numiters.." garbage : ".. collectgarbage("count")/1024 .." mb") 
  end
end
te = sys.clock()
print("Time to read 1000 times was " .. te -ts )
paths.rmall(featurename,"yes")