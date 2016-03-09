--
-- Created by IntelliJ IDEA.
-- User: petrfiala
-- Date: 18/01/16
-- Time: 14:41
-- To change this template use File | Settings | File Templates.
--


function dirLookup(dir)
    local listOfFiles = {}
    local p = io.popen('find "' .. dir .. '" -type f') --Open directory look for files, save data in p. By giving '-type f' as parameter, it returns all files.
    for file in p:lines() do --Loop through all files
    listOfFiles[#listOfFiles + 1] = file
    print(file)
    end
    return listOfFiles
end

function file_exists(name)
    local f = io.open(name, "r")
    if f ~= nil then io.close(f) return true else return false end
end

function preTrainModel()

    if (not file_exists(paths.concat(opt.save, 'preTrainedModel.net'))) then
        -- create temp model for pretraining
        model = nn.Sequential()
        model:add(glimpse)
        model:add(classifier)

        parameters, gradParameters = model:getParameters()

        local A, B, a, b = 0, WIDTH, -1, 1
        wMin = ((glimpseSize / 2) - A) * (b - a) / (B - A) + a
        wMax = ((DATA_WIDTH - glimpseSize / 2) - A) * (b - a) / (B - A) + a

        local A, B, a, b = 0, HEIGHT, -1, 1
        hMin = ((glimpseSize / 2) - A) * (b - a) / (B - A) + a
        hMax = ((DATA_HEIGHT - glimpseSize / 2) - A) * (b - a) / (B - A) + a

        print("wMin: " .. wMin .. " X wMax: " .. wMax)
        print("hMin: " .. hMin .. " X hMax: " .. hMax)

        print("==> PreTraining")
        while epoch < opt.preTrainEpochs do
            preTrainingOptim()
        end

        -- PRE-TRAINED
        torch.save(paths.concat(opt.save, 'preTrainedModel.net'), model)
    else
        local tempModel = torch.load(paths.concat(opt.save, 'preTrainedModel.net'))
        glimpse = tempModel:get(1):clone()
        --        fixedClassifier = tempModel:get(2):clone()
    end

    printOutKernels()
    opt.preTrain = false


    model = agent
    parameters, gradParameters = model:getParameters()

    -- add reinforce loss
    dofile '3_loss.lua'
    epoch = 0
end


function printOutKernels()
    if (conv) then
        local weights = conv:get(1).weight
        for i = 1, glimpseCount do
            local img = image.toDisplayTensor { input = weights[{ {}, i, {}, {} }], padding = 1, scaleeach = 20 }
            image.save(paths.concat(opt.save, 'ch_' .. i .. 'w_epoch_' .. epoch .. '.png'), img)
        end
    end
end
