--
-- Created by IntelliJ IDEA.
-- User: petrfiala
-- Date: 15/01/16
-- Time: 10:15
-- To change this template use File | Settings | File Templates.
--

require 'dp'
--require 'dpnn'
require 'rnn'
require 'image'

----------------------------------------------------------------------
print '==> define parameters of the model'

-- 10-class problem
noutputs = #classes

-- input dimensions
nfeats = N_CHANNELS
width = WIDTH
height = HEIGHT

-- location sensor
lsSize = 128

-- glimpse sensor
glimpseSize = 8
glimpseCount = 2
glimpseScale = 2

gsSize = 128

-- hidden units, filter sizes (for ConvNet only):
nstates = { 36 }
filtsize = { 3 }
poolsize = { 2 }
remainSize = 3

-- glimpse
gSize = 256

-- recurrent
rSize = 256
rho = opt.steps * (opt.digits)      -- + 1 for ending class

-- action location
locatorStd = 0.11
stochastic = false
unitPixels = WIDTH - glimpseSize   -- center of the smallest glimpses will touch the border of the image
rewardScale = 1

----------------------------------------------------------------------
print '==> construct model'

--[[ LOCATION SENSOR ]]--
locationSensor = nn.Sequential()
locationSensor:add(nn.SelectTable(2))
locationSensor:add(nn.Linear(2, lsSize))
locationSensor:add(nn.ReLU())



--[[ GLIMSE SENSOR ]]--
glimpseSensor = nn.Sequential()
glimpseSensor:add(nn.DontCast(nn.SpatialGlimpse(glimpseSize, glimpseCount, glimpseScale):float(),true))
-- Convolution
conv = nn.Sequential()
conv:add(nn.SpatialConvolution(glimpseCount * nfeats, nstates[1], filtsize[1], filtsize[1]))
conv:add(nn.ReLU())
conv:add(nn.SpatialMaxPooling(poolsize[1], poolsize[1]))
conv:add(nn.Reshape(nstates[1] * (remainSize^2)))
conv:add(nn.Linear(nstates[1] * (remainSize^2), gsSize))
glimpseSensor:add(conv)
-- MLP
--mlp = nn.Sequential()
--mlp:add(nn.Collapse(3))
--mlp:add(nn.Linear(nfeats * (glimpseSize^2) * glimpseCount, gsSize))
--glimpseSensor:add(mlp)
glimpseSensor:add(nn.ReLU())



--[[ GLIMPSE ]]--
glimpse = nn.Sequential()
glimpse:add(nn.ConcatTable():add(locationSensor):add(glimpseSensor))
glimpse:add(nn.JoinTable(1,1))
glimpse:add(nn.Linear(lsSize + gsSize, gSize))
glimpse:add(nn.ReLU())
glimpse:add(nn.Linear(gSize, rSize))



--[[ RECURRENT ]]--
recurrent = nn.Linear(rSize, rSize)
rnn = nn.Recurrent(rSize, glimpse, recurrent, nn.ReLU(), 99999)


--[[ ACTION LOCATION ]]--
locator = nn.Sequential()
locator:add(nn.Linear(rSize, 2))
locator:add(nn.HardTanh()) -- bounds mean between -1 and 1
--morn = MReinforceNormal(2 * locatorStd, stochastic) -- sample from normal, uses REINFORCE learning rule
--locator:add(morn)
locator:add(nn.ReinforceNormal(2*locatorStd, stochastic))
assert(locator:get(3).stochastic == stochastic, "Please update the dpnn package : luarocks install dpnn")
locator:add(nn.HardTanh()) -- bounds sample between -1 and 1
locator:add(nn.MulConstant(unitPixels / width))

--[[ ATTENTION MODEL ]]--
dofile 'RecurrentAttention.lua'
attention = RecurrentAttention(rnn, locator, rho, {rSize})


--[[ AGENT ]]--
agent = nn.Sequential()
--agent:add(nn.Convert(ds:ioShapes(), 'bchw'))
agent:add(attention)

--[[ CLASSIFIER ]]--    test don't have batch
stepsSelection = nn.ConcatTable()
for d = 1, opt.digits do
    stepsSelection:add(nn.SelectTable(d * opt.steps))
end
agent:add(stepsSelection)

classifier = nn.Sequential()
classifier:add(nn.Linear(rSize, noutputs))
classifier:add(nn.LogSoftMax())

agent:add(nn.Sequencer(classifier))

--[[ REWARD PREDICTOR ]]--
seq = nn.Sequential()
seq:add(nn.Constant(1,1))
seq:add(nn.Add(1))
concat = nn.ConcatTable():add(nn.Identity()):add(nn.Sequencer(seq))
concat2 = nn.ConcatTable():add(nn.Identity()):add(concat)

-- output will be : {classpred, {classpred, basereward}}
agent:add(concat2)

model = agent



















