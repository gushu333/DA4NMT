import torch
from torch.autograd import Variable
def main():
	dtype = torch.FloatTensor

	N, D_in, H, H_out = 64, 1000, 100, 10

	x = Variable(torch.randn(N, D_in).type(dtype), requires_grad = False)
	y = Variable(torch.randn(N, H_out).type(dtype), requires_grad = False)

	w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad = True)
	w2 = Variable(torch.randn(H, H_out).type(dtype), requires_grad = True)

	learning_rate = 1e-6

	for i in range(200):
		y_pred = x.mm(w1).clamp(min=0).mm(w2)
		loss = (y_pred - y).pow(2).sum()
		print(i, loss.data[0])
		loss.backward()
		w1.data -= learning_rate * w1.grad.data
		w2.data -= learning_rate * w2.grad.data

		w1.grad.data.zero_()
		w2.grad.data.zero_()




if __name__ == "__main__":
	print 123
	main()