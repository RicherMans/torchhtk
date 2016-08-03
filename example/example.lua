local htkutils = paths.dofile('../init.lua')
require "sys"

local size = 120
local dim = 1209
arr = torch.FloatTensor(size,dim)
for i=1,size do
    arr[i]:fill(i)
end

local featurename = "feature.plp"

ts = sys.clock()
htkutils.write(featurename,arr,"PLP_0")
te = sys.clock()
print ("Time to write was " .. te - ts )

local sample
ts = sys.clock()
sample = htkutils.loadsample(featurename,1)
te = sys.clock()
print (string.format("Time to load 1 sample was %.6f s",te - ts ))

ts = sys.clock()
for i=1,size do
	sample = htkutils.loadsample(featurename,i)
end
te = sys.clock()
print (string.format("Time to load %i samples was %.6f s",size, te - ts ))

local feat
ts = sys.clock()
feat = htkutils.load(featurename)
te = sys.clock()
print(string.format("Time to load feature was %.6f s" ,te -ts ))

ts= sys.clock()
for i=1,1000 do
  feat = htkutils.load(featurename)
  collectgarbage()
end
te = sys.clock()
print("Time to read 1000 times was " .. te -ts )



ts =sys.clock()
header = htkutils.loadheader(featurename)
te =sys.clock()
print("Time to read header was " .. te -ts)

sys.fexecute('rm '..featurename)


-- ts=sys.clock()

-- featuretable = {}
-- for i=1,10 do
--     featuretable[i] = featurename
-- end


