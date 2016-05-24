local htkutils = paths.dofile('../init.lua')
require "sys"

local size = 50000
local dim = 4
arr = torch.FloatTensor(size,dim)
for i=1,size do
    arr[i]:fill(i)
end

ts = sys.clock()
htkutils.write("feature.plp",arr,"PLP_0")
te = sys.clock()
sys.fexecute('rm feature.plp')
print ("Time to write was " .. te - ts )


ts= sys.clock()
for i=1,1000 do
  feat = htkutils.load('example/feature.plp')
  collectgarbage()
end
te = sys.clock()
print("Time to read was " .. te -ts )



ts =sys.clock()
header = htkutils.loadheader('example/feature.plp')
te =sys.clock()
print("Time to read header was " .. te -ts)

-- ts=sys.clock()

-- featuretable = {}
-- for i=1,10 do
--     featuretable[i] = "feature.plp"
-- end


