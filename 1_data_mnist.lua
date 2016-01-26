--
-- Created by IntelliJ IDEA.
-- User: petrfiala
-- Date: 15/01/16
-- Time: 10:15
-- To change this template use File | Settings | File Templates.
--

-- classes
classes = { '1', '2', '3', '4', '5', '6', '7', '8', '9', '0' }
WIDTH = 28
HEIGHT = 28
DATA_N_CHANNEL = 1
ninputs = WIDTH * HEIGHT

---------------------------------------------------------------------------------
print("==> Loading data")

local dataset = (opt ~= nil and opt.dataset or 'mnist')

trainData, testData = {}, {}


if dataset == 'mnist' then
    local temp = torch.load('data/mnist/train.th7', 'ascii')
    trainData.data = temp[1]
    trainData.labels = temp[2]
    temp = torch.load('data/mnist/test.th7', 'ascii')
    testData.data = temp[1]
    testData.labels = temp[2]
end


---------------------------------------------------------------------------------
print("==> Preprocessing data")

trainData.data = trainData.data:double()
testData.data = testData.data:double()

trainData.data = trainData.data:transpose(2,3):transpose(2,4)
testData.data = testData.data:transpose(2,3):transpose(2,4)


print("==> Preprocessing normalization")

local mean = trainData.data:mean()
local std = trainData.data:std()

trainData.data = trainData.data:add(-mean):div(std)
testData.data = testData.data:add(-mean):div(std)

trsize = trainData.labels:size(1)
tesize = testData.labels:size(1)

for i=1,trsize do
    trainData.labels[i] = (trainData.labels[i] == 0 and 10 or trainData.labels[i])
end

for i=1,tesize do
    testData.labels[i] = (testData.labels[i] == 0 and 10 or testData.labels[i])
end

print(trainData.data[trsize])

print(testData.data[tesize])





