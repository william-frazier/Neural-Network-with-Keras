# run NN code

from dataFP import load_wine_data
import FP_nnet

print("loading Kaggle dataset")
(train_data, valid_data, test_data, data_stream) = load_wine_data()

# reduce data sets for faster speed:
train_data = train_data
valid_data = valid_data



net = FP_nnet.Network([6, 100, 3])

print("training")

net.train(train_data, valid_data, epochs=10, mini_batch_size=8, alpha=0.5)

ncorrect = net.evaluate(valid_data)
print("Validation accuracy: %.3f%%" % (100 * ncorrect / len(valid_data)))
