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

function MultiObjectRewardCriterion:__init(module, scale, criterion, nGlimpses)
    parent.__init(self)
    self.module = module -- so it can call module:reinforce(reward)
    self.scale = scale or 1 -- scale of reward
    self.criterion = criterion or nn.MSECriterion() -- baseline criterion
    self.sizeAverage = true
    self.gradInput = { torch.Tensor() }
    self.nGlimpses = nGlimpses
end

function MultiObjectRewardCriterion:updateOutput(input, target)
    assert(torch.type(input) == 'table')
    local input = input[1]

    self._nObjects = #input
    self.reward = {}
    self.output = 0

    for object = 1, self._nObjects do
        local objectInput = input[object]
        local objectTarget = target[object]

        self._maxVal = self._maxVal or objectInput.new()
        self._maxIdx = self._maxIdx
                or torch.type(objectInput) == 'torch.CudaTensor' and objectInput.new()
                or torch.LongTensor()

        -- max class value is class prediction
        torch.max(self._maxVal, self._maxIdx, objectInput, 2) -- torch.max([resval, resind,] x [,dim])

        if torch.type(self._maxIdx) ~= torch.type(objectTarget) then
            self._target = self._target or self._maxIdx.new()
            self._target:resize(objectTarget:size()):copy(objectTarget)
            objectTarget = self._target
        end

        -- reward = scale when correctly classified
        self._reward = self._maxIdx.new()
        self._reward:eq(self._maxIdx, objectTarget)
        self.reward[object] = self.reward[object] or objectInput.new()
        self.reward[object]:resize(self._reward:size(1), self._reward:size(2)):copy(self._reward)
        self.reward[object]:mul(self.scale)

        -- backprop only error for record in batch where prev digit was correct
        --    self._mask = self._reward:clone()

        -- loss = -sum(reward)
        self.output = self.output - self.reward[object]:sum()
    end
    if self.sizeAverage then
        self.output = self.output / (self._nObjects * input[1]:size(1))     -- div by # of objects and batch size
    end

    return self.output
end

function MultiObjectRewardCriterion:updateGradInput(inputTable, target)
    assert(torch.type(inputTable) == 'table')
    local input = inputTable[1]
    local baseline = inputTable[2]

    self._nObjects = #input
    self.gradInput, self.vrReward = {}, {}
    self.gradInput[1] = {}
    self.gradInput[2] = {}
    for glimpse =1, self._nObjects * self.nGlimpses do
        self.vrReward[glimpse] = self.vrReward[glimpse] or self.reward[1].new()
        self.vrReward[glimpse]:resize(self.reward[1]:size(1)):zero()
    end

    for object = 1, self._nObjects do

        local objectInput = input[object]
        local objectBaseline = baseline[object]
        local objectReward = self.reward[object]

        -- reduce variance of reward using baseline and copy reward into the n-th step glimpses
        self.vrReward[object * self.nGlimpses] = self.vrReward[object] or objectReward.new()
        self.vrReward[object * self.nGlimpses]:resize(self.reward[object]:size(1)):copy(objectReward)
        self.vrReward[object * self.nGlimpses]:add(-1, objectBaseline)
        if self.sizeAverage then
            self.vrReward[object * self.nGlimpses]:div(objectInput:size(1))
        end

        -- zero gradInput (this criterion has no gradInput for class pred)
        self.gradInput[1][object] = torch.Tensor()
        self.gradInput[1][object]:resizeAs(objectInput):zero()
        -- learn the baseline reward
        self.gradInput[2][object] = torch.Tensor()
        self.gradInput[2][object] = self.criterion:backward(objectBaseline, objectReward)
    end

    -- broadcast reward to modules
    self.module:reinforce(self.vrReward)
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
