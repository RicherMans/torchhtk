local ffi = require("ffi")
local argcheck = require 'argcheck'

ffi.cdef[[
          typedef struct { int nsamples, sample_period; short samplesize, parmkind; } htkheader_t;
          int readhtkheader(const char* fname,htkheader_t *header);
          int readhtkfile(const char* fname,THFloatTensor* output);
          int writehtkfile(const char* fname,htkheader_t *header,THFloatTensor* data);
          int readhtksample(const char* fname,int sample,THFloatTensor* output);
          ]]

local cflua = ffi.load(package.searchpath('libtorchhtk', package.cpath,package.path))


local htkutils = {}

local htkheader=nil
-- Metatable for the htk header
local mt = {
  __len = function(a) return a.nsamples end,
  __tostring = function(a) return "NSamples: " .. a.nsamples .. "\nSamplesize: " .. a.samplesize .. "\nSample_Period: " .. a.sample_period end,
}
htkheader = ffi.metatype("htkheader_t", mt)

local loadheader_check = argcheck{
  help=[[
     Loads an htk header without loading in the data.
     Tested only on Linux (as it uses command-line linux utilities to scale up)
    ]],
  {
    name='filename',
    type='string',
    help="The file to be read",
    check=function(fname)
      if paths.filep(fname) then return true end
    end,
  }
}


function htkutils.loadheader(...)
    local filename = loadheader_check(...)
    local header = ffi.new("htkheader_t")
    local res = cflua.readhtkheader(filename,header)
    -- Samplesize is in float
    header.samplesize = header.samplesize/4
    if res == 0 then
        return header
    else
        error("Cannot load header for file "..filename)
    end
end


local loadhtkfile_check = argcheck{
  {
    name='filename',
    type='string',
    help="The file to be read",
    check=function(fname)
      if paths.filep(fname) then return true end
    end,
  }

}
-- Loads in an htk feature and returns the given data only!
-- If header needs ot be used, refer to loadheader
function htkutils.load(...)
  local filename = loadhtkfile_check(...)
  local out = torch.FloatTensor()
  local res = cflua.readhtkfile(filename,out:cdata())
  assert(res == 0, "Something went wrong while reading feature "..filename)
  return out
end

local loadhtksample_check = argcheck{
  {
    name='filename',
    type='string',
    help="The file to be read",
    check=function(fname)
      if paths.filep(fname) then return true end
    end,
  },
  {
    name="sample",
    type="number",
    help="The sample which needs to be loaded",
    check = function(sample)
      if sample > 0 then return true else return false end
    end
  },
  {
    name="buf",
    type='torch.FloatTensor',
    opt=true
  }

}

function htkutils.loadsample(...)
  local filename, sample, buf = loadhtksample_check(...)
  local out = buf or torch.FloatTensor()
  local res = cflua.readhtksample(filename,sample,out:cdata())
  -- assert(res == 0, "Something went wrong while reading feature "..filename .. " ( probably sample is out of range )")
  return out,res
end


local FEATTYPES={}
FEATTYPES["LPC"] = 1
FEATTYPES["LPCREFC"] = 2
FEATTYPES["LPCEPSTRA"] = 3
FEATTYPES["LPCDELCEP"] = 4
FEATTYPES["IREFC"] = 5
FEATTYPES["MFCC"] = 6
FEATTYPES["FBANK"] = 7
FEATTYPES["MELSPEC"] = 8
FEATTYPES["USER"] = 9
FEATTYPES["DISCRETE"] = 10
FEATTYPES["PLP"] = 11

local MODIFIER={}
MODIFIER["E"] = 64 -- has energy
MODIFIER["N"] = 128 -- absolute energy supressed
MODIFIER["D"] = 256 -- has delta coefficients
MODIFIER["A"] = 512 -- has acceleration (delta-delta) coefficients
MODIFIER["C"] = 1024 -- is compressed
MODIFIER["Z"] = 2048 -- has zero mean static coefficients
MODIFIER["K"] = 4096 -- has CRC checksum
MODIFIER["0"] = 8192 -- has 0th cepstral coefficient
MODIFIER["V"] = 16384 -- has VQ data
MODIFIER["T"] = 32768 -- has third differential coefficients

local TABLETYPES = ""
for i in pairs(FEATTYPES) do
  TABLETYPES = TABLETYPES .. " " .. i
end

local writehtkfile_check = argcheck{
  {
    name='filename',
    help="The file to be written",
    type='string',
    check=function(fname)
    -- can only be used with a new file
      if paths.filep(fname) then return false else return true end
    end,
  },
  {
    name='data',
    help="The data to be written. We expect it to be of size nsamples * dim.",
    type='torch.FloatTensor'
  },
  {
    name='parmkind',
    help="The parameters type, needs to be one of".. TABLETYPES,
    type='string',
    check = function (kind)
      local splits = kind:split("_")
      if FEATTYPES[splits[1]] ~= nil then
        return true
      end
      return false
    end,
  },
  {
    name = 'sampleperiod',
    default=100000,
    help="Sampleperiod is given in nano seconds.",
    type='number',
    check = function (num)
      if num > 0 then return true end
    end
  }
}

local function parseparmkind(kind)
  local splits = kind:split("_")
  local startkind = FEATTYPES[splits[1]]
  for i=2,#splits do
    assert(MODIFIER[splits[i]],string.format("Modifier %s could not be parsed",splits[i]))
    startkind = startkind + MODIFIER[splits[i]]
  end
  return startkind
end

function htkutils.write(...)
  local filename, data, paramkind, sampleperiod = writehtkfile_check(...)
  assert(data:dim() == 2 , "Data needs to be two dimensional, with dimensions nsamples * featuredim.")
  local header = ffi.new("htkheader_t")
  -- Featdimension
  header.samplesize = data:size(2) * 4
  -- Number of samples
  header.nsamples = data:size(1)
  -- Paramkind parsing
  header.parmkind = parseparmkind(paramkind)
  -- Sample period is for most usecases not important
  header.sample_period = sampleperiod
  cflua.writehtkfile(filename,header,data:cdata())
end

return htkutils
