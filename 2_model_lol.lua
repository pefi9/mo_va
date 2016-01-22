--
-- Created by IntelliJ IDEA.
-- User: petrfiala
-- Date: 15/01/16
-- Time: 10:15
-- To change this template use File | Settings | File Templates.
--

require 'dp'
require 'dpnn'
require 'rnn'
require 'nn'
require 'image'

----------------------------------------------------------------------
print '==> define parameters of the model'

-- 10-class problem
noutputs = #classes

-- input dimensions
nfeats = 1
width = WIDTH
height = HEIGHT

-- hidden units, filter sizes (for ConvNet only):
nstates = { 8, 16, 128 }
filtsize = { 5, 3 }
poolsize = { 2, 2 }
remainSize = 3
normkernel = image.gaussian1D(7)


----------------------------------------------------------------------
print '==> construct model'

model = nn.Sequential()

-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolution(nfeats, nstates[1], filtsize[1], filtsize[1]))
model:add(nn.ReLU())
model:add(nn.SpatialLPPooling(nstates[1],2,poolsize[1],poolsize[1]))
model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolution(nstates[1], nstates[2], filtsize[2], filtsize[2]))
model:add(nn.ReLU())
model:add(nn.SpatialLPPooling(nstates[2],2,poolsize[2],poolsize[2]))
model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

-- stage 3 : standard 2-layer neural network
model:add(nn.Reshape(nstates[2] * remainSize * remainSize))
model:add(nn.Linear(nstates[2] * remainSize * remainSize, nstates[3]))
model:add(nn.ReLU())
model:add(nn.Linear(nstates[3], noutputs))

-- This loss requires the outputs of the trainable model to
-- be properly normalized log-probabilities, which can be
-- achieved using a softmax function

model:add(nn.LogSoftMax())
