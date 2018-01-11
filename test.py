import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers,Variable,serializers

class LSTM(chainer.Chain):
	def __init__(self, n_in, n_units,n_out, train=True):
		super(LSTM, self).__init__(
			l1=L.Linear(n_in, n_units),
			l2=L.LSTM(n_units,n_units),
			l3=L.Linear(n_units, n_out),
		)

	def __call__(self, x_data,y_data):
		x = chainer.Variable(x_data.astype(np.float32).reshape(len(x_data),1))
		y = chainer.Variable(y_data.astype(np.float32).reshape(len(y_data),1))
		return F.mean_squared_error(self.predict(x),y)


	def predict(self,x):
		h0 = self.l1(x)
		h1 = self.l2(h0)
		y = self.l3(h1)
		return y

	def reset_state(self):
		self.l2.reset_state()


if __name__ == "__main__":
	x = np.linspace(0,10*np.pi,10000)
	t = np.sin(x)

	model = LSTM(1,5,1)
	serializers.load_npz("lstm_model_1_5_1.npz",model)
	model.reset_state()
	for i in range(10000):
		x = chainer.Variable(np.array([[i/300]],dtype=np.float32))
		print(model.predict(x).data)



