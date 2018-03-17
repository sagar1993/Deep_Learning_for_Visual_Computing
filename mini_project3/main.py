import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from helper import *


if __name__ == "__main__":
	x_train, y_train, x_test, y_test = generate_data()

	print("CNN")
	
	model, loss_l = build_model_cnn(x_train[:600], y_train[:600], 8,8, print_loss=True)

	## training accuracy
	print(np.sum(y_train == predict_cnn(model, x_train[:600])) / len(x_train[:600]))

	## test accuracy
	print(np.sum(y_test == predict_cnn(model, x_test[:400])) / len(x_test[:400]))

	plot("cnn training error", range(8), loss_l, "training error", "iterations","error")



	print("CNN with maxpooling")
	model, loss_l = build_model_cnn_maxpool(x_train[:600], y_train[:600], 8,8, print_loss=True)

	## training accuracy
	print(np.sum(y_train == predict_maxpool(model, x_train[:600])) / len(x_train[:600]))

	## test accuracy
	print(np.sum(y_test == predict_maxpool(model, x_test[:400])) / len(x_test[:400]))

	plot("cnn maxpool training error", range(8), loss_l, "training error", "iterations","error")


	print("CNN with maxpooling and dropout")
	model, loss_l = build_model_maxpool_dropout(x_train[:600], y_train[:600], 8,8, print_loss=True)

	## training accuracy
	print(np.sum(y_train == predict_maxpool_dropout(model, x_train[:600])) / len(x_train[:600]))

	## test accuracy
	print(np.sum(y_test == predict_maxpool_dropout(model, x_test[:400])) / len(x_test[:400]))

	plot("cnn maxpool training error", range(8), loss_l, "training error", "iterations","error")