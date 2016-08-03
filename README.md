# torch-htk

This project provides a simple api to load and save [HTK](http://htk.eng.cam.ac.uk/) features. 

This library is so far only tested on linux, thus there is no guarantee for Wind and OSX!

## Prequisites

* [Torch](http://torch.ch/) is the only prequisite for this library. 

## Installation

The most convinient way is to reach this repository over [luarocks](https://luarocks.org/). 

```
luarocks install torchhtk
```

If that is not possible, do it manually:

```
git clone https://github.com/RicherMans/torchhtk
cd torchhtk
luarocks make rocks/torchhtk-0.0-1.rockspec
```

## Methods and classes

 * [header] loadheader( `filepath` [string]) :  Loads only the header of the specified `filepath`. No extra memory is allocated or data loaded.
 * [torch.FloatTensor] load(`filepath` [string]) : Loads the data from the given filepath. Returns a torch.FloatTensor which dimensions are `nsamples * featuredim`.
 * write( `filepath` [string], `data` [torch.FloatTensor], `featuretype` [string]) : Writes out an htk file. `data` specifies the vector which should be written out with sizes `nsampels * featuredim`. `featuretype` needs to be one of the known [HTK types](http://www.ee.columbia.edu/ln/LabROSA/doc/HTKBook21/node58.html), e.g. `PLP_0` or for any default data, simply use `USER`.
 * loadsample(`filepath`[string], sample [number]) Loads only a given sample from the provided file. This method is generally less efficient than ```load```, if one wants to load in the whole file at once, but more efficient if only single samples are needed.

## Usage

To read HTK files simply use the provided API.

```lua
local htkutils = require 'torchhtk'
-- Only loads the header of a file
local header = htkutils.loadheader('feature.plp')
print(header) -- prints number of samples and other information
-- Returns a torch.FloatTensor()
local tensor = htkutils.load('feature.plp')
-- Loads only the first sample in (instead of the whole utterance)
local sample = htkutils.loadsample('feature.plp',1)
```

To write files ( note that we only support `torch.FloatTensor` ):

```lua
require 'torch'
local htkutils = require 'torchhtk'

local size = 50000
local dim = 4
local tensor = torch.FloatTensor(size,dim)
for i=1,size do
    tensor[i]:fill(i)
end
-- Outputfile should be named feature.plp
-- Featuretype is set to be PLP
htkutils.write("feature.plp",tensor,"PLP")

```




