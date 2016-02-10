--
-- Created by IntelliJ IDEA.
-- User: petrfiala
-- Date: 15/01/16
-- Time: 10:15
-- To change this template use File | Settings | File Templates.
--

-- classes
classes = { '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'end' }
DATA_WIDTH = 28
DATA_HEIGHT = 28
WIDTH = DATA_WIDTH * opt.digits + 20
HEIGHT = DATA_HEIGHT
DATA_N_CHANNEL = 1
ninputs = WIDTH * HEIGHT

---------------------------------------------------------------------------------
print("==> Loading data")

local dataset = (opt ~= nil and opt.dataset or 'mnist')

trainData, testData = {}, {}


if dataset == 'mnist' then

    -- train data
    local temp = torch.load('data/mnist/train.th7', 'ascii')
    trsize = temp[1]:size()[1]

    trainData.data = torch.DoubleTensor(trsize, HEIGHT, WIDTH, 1)
    trainData.labels = torch.DoubleTensor(trsize, opt.digits + 1)
    for rec = 1, trsize do
        local tempData
        for digit = 1, opt.digits do
            if digit == 1 then
                tempData = temp[1][rec]
                trainData.labels[rec][digit] = (temp[2][rec] == 0 and 10 or temp[2][rec])
            else
                tempData = tempData:cat(torch.FloatTensor(HEIGHT, 20):fill(0), 2)     -- create space between digits
                local rand = math.floor(math.random() * trsize) + 1
                tempData = tempData:cat(temp[1][rand], 2)
                trainData.labels[rec][digit] = (temp[2][rand] == 0 and 10 or temp[2][rand])
            end
        end
        trainData.data[rec] = tempData
        trainData.labels[rec][opt.digits + 1] = 11
    end

    -- test data
    local temp = torch.load('data/mnist/test.th7', 'ascii')
    tesize = temp[1]:size()[1]

    testData.data = torch.DoubleTensor(tesize, HEIGHT, WIDTH, 1)
    testData.labels = torch.DoubleTensor(tesize, opt.digits + 1)
    for rec = 1, tesize do
        local tempData
        for digit = 1, opt.digits do
            if digit == 1 then
                tempData = temp[1][rec]
                testData.labels[rec][digit] = (temp[2][rec] == 0 and 10 or temp[2][rec])
            else
                tempData = tempData:cat(torch.FloatTensor(HEIGHT, 20):fill(0), 2)     -- create space between digits
                local rand = math.floor(math.random() * tesize) + 1
                tempData = tempData:cat(temp[1][rand], 2)
                testData.labels[rec][digit] = (temp[2][rand] == 0 and 10 or temp[2][rand])
            end
        end
        testData.data[rec] = tempData
        testData.labels[rec][opt.digits + 1] = 11
    end

end

---------------------------------------------------------------------------------
print("==> Preprocessing data")

trainData.data = trainData.data:transpose(2, 3):transpose(2, 4)
testData.data = testData.data:transpose(2, 3):transpose(2, 4)


print("==> Preprocessing normalization")

--local mean = trainData.data:mean()
--local std = trainData.data:std()

--trainData.data = trainData.data:add(-mean):div(std)
--testData.data = testData.data:add(-mean):div(std)

local max = trainData.data:max()
trainData.data = trainData.data:div(max)
testData.data = testData.data:div(max)

--print(trainData.data[trsize])
--
--print(testData.data[tesize])





