local MOSelectTable, parent = torch.class('MOSelectTable', 'nn.SelectTable')

function MOSelectTable:__init(index)
    parent.__init(self, index)
    self.reward = {}
end

function MOSelectTable:reinforce(reward)
    if self.index < 0 then
        error "Reinforce is not implemented for negative index value"
    else
        self.reward[self.index] = reward
    end

    for idx = self.index - 1, 1, -1 do
        self.reward[idx] = torch.Tensor()
        self.reward[idx]:resizeAs(reward):zeros()
    end

    return self.reward
end

function MOSelectTable:type(type, tensorCache)
    self.reward = {}
    return parent.type(self, type, tensorCache)
end
