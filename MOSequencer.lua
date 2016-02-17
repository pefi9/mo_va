------------------------------------------------------------------------
--[[ MOSequencer ]] --
-- Encapsulates a Module.
-- Input is a sequence (a table) of tensors.
-- Output is a sequence (a table) of tensors of the same length.
-- Applies the module to each element in the sequence.
-- Handles both recurrent modules and non-recurrent modules.
-- The sequences in a batch must have the same size.
-- But the sequence length of each batch can vary.
------------------------------------------------------------------------
local MOSequencer, parent = torch.class('MOSequencer', 'nn.Sequencer')

function MOSequencer:__init(module, nSteps)
    parent.__init(self, module)
    self.nSteps = nSteps
end

function MOSequencer:updateGradInput(inputTable, gradOutputTable)
    assert(torch.type(gradOutputTable) == 'table', "expecting gradOutput table")
    assert(#gradOutputTable == #inputTable, "gradOutput should have as many elements as input")

    -- back-propagate through time (BPTT)
    self.gradInput = {}
    for step = #gradOutputTable, 1, -1 do
        if(self.moReward) then
            local indx = math.ceil(step / self.nSteps)
            parent.reinforce(self, self.moReward[indx])
        end

        self.gradInput[step] = self.module:updateGradInput(inputTable[step], gradOutputTable[step])
    end

    assert(#inputTable == #self.gradInput, #inputTable .. " ~= " .. #self.gradInput)

    return self.gradInput
end

function MOSequencer:reinforce(reward)
    if reward:dim() == 2 then
        self.moReward:resizeAs(reward):copy(reward)
    else
        return parent.reinforce(self, reward)
    end
end

MOSequencer.__tostring__ = nn.Decorator.__tostring__