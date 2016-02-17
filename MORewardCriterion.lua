--
-- Created by IntelliJ IDEA.
-- User: petrfiala
-- Date: 16/02/2016
-- Time: 16:05
-- To change this template use File | Settings | File Templates.
--
------------------------------------------------------------------------
--[[ VRClassReward ]] --
-- Variance reduced classification reinforcement criterion.
-- input : {class prediction, baseline reward}
-- Reward is 1 for success, Reward is 0 otherwise.
-- reward = scale*(Reward - baseline) where baseline is 2nd input element


-- concat = nn.ConcatTable():add(nn.Select(n)):add(nn.Select(n*2))...:add(nn.Select(-1))

-- [[Note : for RNNs with R = 1 for last step in sequence, encapsulate it
-- in nn.ModuleCriterion(VRClassReward, nn.SelectTable(-1))]]
------------------------------------------------------------------------
local MultiObjectRewardCriterion, parent = torch.class("MultiObjectRewardCriterion", "nn.Criterion")

function MultiObjectRewardCriterion:__init(module, scale, criterion)
    parent.__init(self)
    self.module = module -- so it can call module:reinforce(reward)
    self.scale = scale or 1 -- scale of reward
    self.criterion = criterion or nn.MSECriterion() -- baseline criterion
    self.sizeAverage = true
    self.gradInput = { torch.Tensor() }
end

function MultiObjectRewardCriterion:updateOutput(input, target)
    assert(torch.type(input) == 'table')
    local input = input[1]
    input = nn.Sequential():add(nn.JoinTable(1)):add(nn.View(input[1]:size())):forward(input)

    self._maxVal = self._maxVal or input.new()
    self._maxIdx = self._maxIdx
            or torch.type(input) == 'torch.CudaTensor' and input.new()
            or torch.LongTensor()

    -- max class value is class prediction
    torch.max(self._maxVal, self._maxIdx, input, 3) -- torch.max([resval, resind,] x [,dim])

    if torch.type(self._maxIdx) ~= torch.type(target) then
        self._target = self._target or self._maxIdx.new()
        self._target:resize(target:size()):copy(target)
        target = self._target
    end

    -- reward = scale when correctly classified
    self._reward = self._maxIdx.new()
    self._reward:eq(self._maxIdx, target)
    self.reward = self.reward or input.new()
    self.reward:resize(self._reward:size(1), self._reward:size(2)):copy(self._reward)
    self.reward:mul(self.scale)

    -- loss = -sum(reward)
    self.output = -self.reward:sum()
    if self.sizeAverage then
        self.output = self.output / (input:size(1) * input:size(2))
    end
    return self.output
end

function MultiObjectRewardCriterion:updateGradInput(inputTable, target)
    --    assert(inputTable[1]:dim() == 3, 'Input[1] has to be in form: #objects x batchSize x #classes')
    --    local input = self:toBatch(inputTable[1], 1)
    --    assert(inputTable[2]:dim() == 3, 'Input[2] has to be in form: #objects x batchSize x 1 (dim of base')
    --    local baseline = self:toBatch(inputTable[2], 1)
    local input = inputTable[1]
    input = nn.Sequential():add(nn.JoinTable(1)):add(nn.View(input[1]:size())):forward(input)
    local baseline = inputTable[2]
    baseline = nn.Sequential():add(nn.JoinTable(1)):add(nn.View(baseline[1]:size())):forward(baseline)

    -- reduce variance of reward using baseline
    self.vrReward = self.vrReward or self.reward.new()
    self.vrReward:resizeAs(self.reward):copy(self.reward)
    self.vrReward:add(-1, baseline)
    if self.sizeAverage then
        self.vrReward:div(input:size(2))
    end
    -- broadcast reward to modules
    self.module:reinforce(self.vrReward)

    -- zero gradInput (this criterion has no gradInput for class pred)
    self.gradInput[1]:resizeAs(input):zero()
    self.gradInput[1] = self:fromBatch(self.gradInput[1], 1)

    -- learn the baseline reward
    self.gradInput[2] = self.criterion:backward(baseline, self.reward)
    self.gradInput[2] = self:fromBatch(self.gradInput[2], 1)
    return self.gradInput
end

function MultiObjectRewardCriterion:type(type)
    self._maxVal = nil
    self._maxIdx = nil
    self._target = nil
    local module = self.module
    self.module = nil
    local ret = parent.type(self, type)
    self.module = module
    return ret
end
