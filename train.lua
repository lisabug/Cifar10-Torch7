require './data_augmentation.lua'
require './preprocessing.lua'
require 'torch'
require './deep_model.lua'
require './model.lua'
require './minibatch_sgd.lua'

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
for k, v in pairs(LABEL2ID) do
    ID2LABEL[v] = k
    CLASSES[v] = k
end

local function label_vector(label_index)
    local vec = torch.Tensor(10):zero()
    vec[math.floor(label_index)] = 1.0
    return vec
end

--function test(model, params, test_x, test_y, classes)
--    local confusion = optim.ConfusionMatrix(classes)
--    local x, y = data_augmentation(test_x, test_y)
--    preprocessing(x, params)
--    step = 64
--    for i = 1, x:size(1), step do
--        if i + step - 1 > x:size(1) then
--            n = x:size(1) - i + 1
--        else
--            n = step
--        end
--        local batch = torch.Tensor(n, x:size(2), x:size(3), x:size(4)):zero()
--        batch:narrow(1,1,n):copy(x:narrow(1,i,n))
--        local preds = model:forward(batch:cuda()):float()
--        for k = 1, preds:size(1) do
--            confusion:add(preds[k], y[i+k-1])
--        end
--        xlua.progress(i, x:size(1))
--    end
--    xlua.progress(x:size(1), x:size(1))
--    return confusion
--end
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

function training()
    local MAX_EPOCH = 100
    local model = deep_model():cuda()
    --local model = cnn_model():cuda()
    --local criterion = nn.ClassNLLCriterion():cuda()
    local criterion = nn.MSECriterion():cuda()
    local sgd_config = {
        learningRate = 1,
        learningRateDecay = 5.0e-6,
        momentum = 0.9,
        xBatchSize = 16
    }
    local params = nil
    local TRAIN_SIZE = 40000
    local TEST_SIZE = 10000
    --local TRAIN_SIZE = 400
    --local TEST_SIZE = 100
    
    
    -- load data
    dofile 'load_data.lua'
    local x = torch.Tensor(trainData.data:size()):copy(trainData.data)
    local y = torch.Tensor(trainData.data:size(1), 10)
    for i = 1, trainData.data:size(1) do
        y[i]:copy(label_vector(trainData.labels[i]))
    end
    local train_x = x:narrow(1, 1, TRAIN_SIZE)
    local train_y = y:narrow(1, 1, TRAIN_SIZE)
    local test_x = x:narrow(1, TRAIN_SIZE+1, TEST_SIZE)
    local test_y = y:narrow(1, TRAIN_SIZE+1, TEST_SIZE)
    --y = torch.Tensor(trainData.labels:size()):copy(trainData.labels)
    

    -- data augmentation
    print("data augmentation ..")
    train_x, train_y = data_augmentation(train_x, train_y)
    collectgarbage()

    print("preprocessing ..")
    params = preprocessing(train_x)
    torch.save("models/preprocess_params.bin", params)
    collectgarbage()

    for epoch = 1, MAX_EPOCH do
        if epoch == MAX_EPOCH then
            sgd_config.learningRate = 0
            sgd_config.learningRateDecay = 0.01
        end

        model:training()
        print("# " .. epoch)
        print("## train")
        --print(minibatch_sgd(model, criterion, train_x, train_y, CLASSES, sgd_config))
        minibatch_sgd(model, criterion, train_x, train_y, CLASSES, sgd_config)
        print("## test")
        model:evaluate()
        print(test(model, params, test_x, test_y, CLASSES))
        epoch = epoch + 1
        torch.save(string.format("models/deep_%d.model", epoch), model)

        collectgarbage()
    end
end


--local cmd = torch.CmdLine()
--cmd:text()
--cmd:text("Cifar-10 Training")
--cmd:text("Options:")
--cmd:text("-seed", 11, 'fixed input seed')
--local opt = cmd:parse(arg)
--print(opt.seed)
torch.manualSeed(11)
training()
