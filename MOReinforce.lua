------------------------------------------------------------------------
--[[ Reinforce ]]--
-- Ref A. http://incompleteideas.net/sutton/williams-92.pdf
-- Abstract class for modules that use the REINFORCE algorithm (ref A).
-- The reinforce(reward) method is called by a special Reward Criterion.
-- After which, when backward is called, the reward will be used to
-- generate gradInputs. The gradOutput is usually ignored.
------------------------------------------------------------------------
local MOReinforce, parent = torch.class("MOReinforce", "nn.Reinforce")

function MOReinforce:__init(stochastic)
    parent.__init(self, stochastic)
end

-- this can be called by updateGradInput
function MOReinforce:rewardAs(input, step)
    step = math.ceil(step / opt.steps)
    self.stepReward = self.reward[step]
    assert(self.stepReward:dim() == 1)
    if input:isSameSizeAs(self.stepReward) then
        return self.stepReward
    else
        if self.stepReward:size(1) ~= input:size(1) then
            -- assume input is in online-mode
            input = self:toBatch(input, input:dim())
            assert(self.stepReward:size(1) == input:size(1), self.stepReward:size(1).." ~= "..input:size(1))
        end
        self._stepReward = self._stepReward or self.stepReward.new()
        self.__stepReward = self.__stepReward or self.stepReward.new()
        local size = input:size():fill(1):totable()
        size[1] = self.stepReward:size(1)
        self._stepReward:view(self.stepReward, table.unpack(size))
        self.__stepReward:expandAs(self._stepReward, input)
        return self.__stepReward
    end
end