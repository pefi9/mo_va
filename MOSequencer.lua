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

function MOSequencer:__init(module)
    parent.__init(self, module)
end

function MOSequencer:reinforce(reward)
    return parent.reinforce(self, reward)
end

MOSequencer.__tostring__ = nn.Decorator.__tostring__