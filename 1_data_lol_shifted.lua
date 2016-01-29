--
-- Created by IntelliJ IDEA.
-- User: petrfiala
-- Date: 18/01/16
-- Time: 13:46
-- To change this template use File | Settings | File Templates.
--

require 'nn'

if (opt == nil) then
    opt = {}
    opt.preTrain = false
end

-- classes
classes = { '1', '2', '3', '4', '5', '6', '7', '8', '9', '0' }

DATA_FILE = '/Users/petrfiala/rts_munity/code/analyzer/data/torch/digits_data.csv'
DATA_WIDTH = 20
DATA_HEIGHT = 30
DATA_N_CHANNEL = 1
WIDTH = 60
HEIGHT = 60
ninputs = WIDTH * HEIGHT
---------------------------------------------------------------------------------
print("==> Loading data")

function getRecord(line)

    line = line:gsub('([\[]+)', '') -- remove '['
    line = line:gsub('([\]]+)', '') -- remove ']'
    line = line:gsub('%s+', "")
    local count = 0
    local record = torch.DoubleTensor(DATA_WIDTH * DATA_HEIGHT + 1)
    for val in line:gmatch('([^, || ^;]+)') do
        count = count + 1
        record[count] = tonumber(val)
    end

    assert(count == (DATA_WIDTH * DATA_HEIGHT + 1), 'Line does not contains precise number of values')
    return record
end


local tempData = {}
local file = io.open(DATA_FILE, 'r')
local line = file:read()

while (line ~= nil) do
    tempData[#tempData + 1] = getRecord(line)
    line = file:read()
end

file:close()

---------------------------------------------------------------------------------
print("==> Preprocessing data")

size = #tempData
trsize = math.floor(size / 10 * 7)
tesize = size - trsize

trainData = {
    data = torch.DoubleTensor(trsize, DATA_N_CHANNEL, HEIGHT, WIDTH),
    labels = torch.Tensor(trsize)
}

testData = {
    data = torch.DoubleTensor(tesize, DATA_N_CHANNEL, HEIGHT, WIDTH),
    labels = {}
}

local shuffle = torch.randperm(size)

local resp = nn.Reshape(DATA_N_CHANNEL, DATA_HEIGHT, DATA_WIDTH)

for i = 1, size do

    -- extend the input image
    local rand_x = (opt.preTrain) and (WIDTH - DATA_WIDTH)/2 or math.random(WIDTH - DATA_WIDTH)
    local rand_y = (opt.preTrain) and (HEIGHT - DATA_HEIGHT)/2 or math.random(HEIGHT - DATA_HEIGHT)

    local temp = tempData[shuffle[i]]:narrow(1, 2, DATA_WIDTH * DATA_HEIGHT) -- get pixels values without label
    temp = resp:forward(temp) -- reshape into 3D tensor
    local tempInput = torch.Tensor(DATA_N_CHANNEL, HEIGHT, WIDTH):fill(0)

    for i = 1, DATA_N_CHANNEL do
--        tempInput[i] = temp[i]
        tempInput[i]:sub(rand_y, rand_y + DATA_HEIGHT - 1, rand_x, rand_x + DATA_WIDTH - 1):copy(temp[i])
    end

    if (i <= trsize) then
        trainData.labels[i] = tempData[shuffle[i]][1]
        trainData.data[i] = tempInput
    else
        testData.labels[i - trsize] = tempData[shuffle[i]][1]
        testData.data[i - trsize] = tempInput
    end
end

---------------------------------------------------------------------------------
print("==> Preprocessing normalization")


local mean = trainData.data:mean()
local std = trainData.data:std()

trainData.data = trainData.data:add(-mean):div(std)
testData.data = testData.data:add(-mean):div(std)

print(trainData.data[trsize])

--local max = trainData.data:max()
--
--trainData.data = trainData.data:div(max)
--testData.data = testData.data:div(max)

for i = 1, trsize do
    trainData.labels[i] = (trainData.labels[i] == 0 and 10 or trainData.labels[i])
end

for i = 1, tesize do
    testData.labels[i] = (testData.labels[i] == 0 and 10 or testData.labels[i])
end
