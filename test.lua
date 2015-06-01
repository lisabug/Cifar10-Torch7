require './data_augmentation.lua'
require './preprocessing.lua'
require './minibatch_sgd.lua'
require './deep_model.lua'
require 'cutorch'
require 'optim'
require 'xlua'
--require './train.lua'

CLASSES = {}
LABEL2ID = {
    airplane = 1,
    automobile = 2,
    bird = 3,
    cat = 4,
    deer = 5,
    dog = 6,
    frog = 7,
    horse = 8,
    ship = 9,
    truck = 10
}
ID2LABEL = {}
for k,v in pairs(LABEL2ID) do
    ID2LABEL[v] = k
    CLASSES[v] = k
end

local function label_vector(label_index)
    local vec = torch.Tensor(10):zero()
    vec[math.floor(label_index)] = 1.0
    return vec
end

function test(model, params, test_x, test_y, classes)
    local confusion = optim.ConfusionMatrix(classes)
    for i = 1, test_x:size(1) do
        local preds = torch.Tensor(10):zero():float()
        local x = data_augmentation(test_x[i], test_y[i])
        local step = 64
        preprocessing(x, params)
        for j = 1, x:size(1), step do
            local batch = torch.Tensor(step, x:size(2), x:size(3), x:size(4)):zero()
            local n = step
            if j + n > x:size(1) then
                n = 1 + n - ((j + n) - x:size(1))
            end
            batch:narrow(1,1,n):copy(x:narrow(1,j,n))
            local z = model:forward(batch:cuda()):float()
            --averaging
            for k = 1, n do
                preds = preds:add(z[k])
            end
        end
        preds:div(x:size(1))
        confusion:add(preds, test_y[i])
        xlua.progress(i, test_x:size(1))
    end
    xlua.progress(test_x:size(1), test_x:size(1))
    return confusion
end


dofile './load_data.lua'
test_x = torch.Tensor(testData.data:size()):copy(testData.data)
test_y = torch.Tensor(testData.data:size(1), 10)
for i = 1, testData.data:size(1) do
    test_y[i]:copy(label_vector(testData.labels[i]))
end

model = torch.load("./models/deep_59.model"):cuda()
params = torch.load("./models/preprocess_params.bin")

print(test(model, params, test_x, test_y, CLASSES))
