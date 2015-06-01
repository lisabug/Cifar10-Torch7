----------------------------------------------------------------------
-- This script downloads and loads the CIFAR-10 dataset
-- http://www.cs.toronto.edu/~kriz/cifar.html
----------------------------------------------------------------------
-- Note: files were converted from their original format
-- to Torch's internal format.

-- The CIFAR-10 dataset provides  3 files:
--    + train: training data
--    + test:  test data

----------------------------------------------------------------------
print '==> loading dataset'

-- We load the dataset from disk, it's straightforward

dataPath = '../../Datasets/Cifar-10/'

trsize = 50000
tesize = 10000

trainData = {
   data = torch.Tensor(trsize, 3*32*32),
   labels = torch.Tensor(trsize),
   size = function() return trsize end
}
for i = 0,4 do
   subset = torch.load(dataPath .. 'data_batch_' .. (i+1) .. '.t7', 'ascii')
   trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
   trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
end
trainData.labels = trainData.labels + 1

subset = torch.load(dataPath .. 'test_batch.t7', 'ascii')
testData = {
   data = subset.data:t():double(),
   labels = subset.labels[1]:double(),
   size = function() return tesize end
}
testData.labels = testData.labels + 1

-- resize dataset (if using small version)
trainData.data = trainData.data[{ {1,trsize} }]
trainData.labels = trainData.labels[{ {1,trsize} }]

testData.data = testData.data[{ {1,tesize} }]
testData.labels = testData.labels[{ {1,tesize} }]

-- reshape data                                                                                     
trainData.data = trainData.data:reshape(trsize,3,32,32)
testData.data = testData.data:reshape(tesize,3,32,32)

print('Training Data:')
print(trainData)
print()

print('Test Data:')
print(testData)
print()

----------------------------------------------------------------------
print '==> visualizing data'

-- Visualization is quite easy, using itorch.image().
if itorch then
   --print('training data:')
   --itorch.image(trainData.data[{ {1,256} }])
   print('test data:')
   itorch.image(testData.data[{ {1,50} }])
   print(testData.labels[{{1,50}}])
end
