--
-- Created by IntelliJ IDEA.
-- User: petrfiala
-- Date: 16/02/2016
-- Time: 16:08
-- To change this template use File | Settings | File Templates.
--

require 'dpnn'

function dpnntest.VRClassReward()
    input = { torch.randn(13, 10), torch.randn(13, 1) }
    target = torch.IntTensor(13):random(1, 10)
    rf = nn.Reinforce()
    vrc = nn.VRClassReward(rf)
    err = vrc:forward(input, target)
    gradInput = vrc:backward(input, target)
    val, idx = input[1]:max(2)
    reward = torch.eq(idx:select(2, 1):int(), target):double()
    err2 = -reward:mean()
    assert(err == err2, "VRClassReward forward err")
    gradInput2 = nn.MSECriterion():backward(input[2], reward)
    assertTensorEq(gradInput[2], gradInput2, 0.000001, "VRClassReward backward baseline err")
    assertTensorEq(gradInput[1], input[1]:zero(), 0.000001, "VRClassReward backward class err")
end

-- # digits = 3, batch size = 13, #classes = 10
function MultiObjectRewardCriterion()
    input = { torch.randn(3, 13, 10), torch.randn(3, 13, 1) }
    target = torch.IntTensor(3, 13):random(1, 10)
    rf = nn.Reinforce()
    morc = MultiObjectRewardCriterion(rf)
    err = morc:forward(input, target)
    gradInput = morc:backward(input, target)
    val, idx = input[1]:max(2)
    reward = torch.eq(idx:select(2, 1):int(), target):double()
    err2 = -reward:mean()
    assert(err == err2, "VRClassReward forward err")
    gradInput2 = nn.MSECriterion():backward(input[2], reward)
    assertTensorEq(gradInput[2], gradInput2, 0.000001, "VRClassReward backward baseline err")
    assertTensorEq(gradInput[1], input[1]:zero(), 0.000001, "VRClassReward backward class err")
end