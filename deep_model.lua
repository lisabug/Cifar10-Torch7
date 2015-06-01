require 'cunn'

function deep_model()
    local model = nn.Sequential()
    
    local input = torch.Tensor(3, 24, 24)

    -- convolution layers
    --model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
    model:add(nn.SpatialConvolutionMM(3, 128, 3, 3, 1, 1))
    model:add(nn.ReLU())

    print(model:forward(input):size())

    --model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
    model:add(nn.SpatialConvolutionMM(128, 256, 3, 3, 1, 1))
    model:add(nn.ReLU())

    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    model:add(nn.Dropout(0.5))

    print(model:forward(input):size())
    
    --model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
    model:add(nn.SpatialConvolutionMM(256, 256, 3, 3, 1, 1))
    model:add(nn.ReLU())
    
    model:add(nn.SpatialConvolutionMM(256, 512, 3, 3, 1, 1))
    model:add(nn.ReLU())

    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    model:add(nn.Dropout(0.5))

    print(model:forward(input):size())

    --model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
    -- 3 3 1 1
    -- 3 3 1 1
    -- no maxpooling
    -- validation 82.7%
    model:add(nn.SpatialConvolutionMM(512, 1024, 3, 3, 1, 1))
    --model:add(nn.ReLU())

    --print(model:forward(input):size())

    --model:add(nn.SpatialConvolutionMM(1024, 1024, 3, 3, 1, 1))
    model:add(nn.ReLU())
    model:add(nn.Dropout(0.5))

    print(model:forward(input):size())	

    model:add(nn.SpatialConvolutionMM(1024, 1024, 1, 1, 1, 1))
    model:add(nn.ReLU())

    model:add(nn.SpatialConvolutionMM(1024, 1024, 1, 1, 1, 1))
    model:add(nn.ReLU())

    model:add(nn.SpatialConvolutionMM(1024, 10, 1, 1, 1, 1))

    model:add(nn.Reshape(10))
    model:add(nn.SoftMax())

    return model
end
