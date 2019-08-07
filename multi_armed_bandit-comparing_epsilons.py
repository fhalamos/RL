#Based on https://github.com/lazyprogrammer/machine_learning_examples/blob/master/rl/comparing_epsilons.py

import numpy as np
import matplotlib.pyplot as plt

class Bandit():

  #Create bandit, input is its true mean
	def __init__(self, m):
		#true mean
		self.m = m
		#mean when experimenting
		self.mean = 0
		#number of times experimented
		self.N = 0

	#Get bandits reward, which is random centered on its mean
	def pull(self):
		return self.m + np.random.randn()

	#After new reward, update mean experimented and number of experiments
	def update(self, x):
		self.mean = (self.mean * self.N + x)/(self.N+1)
		self.N +=1

#Create 3 bandits, each with means m1, m2, m3. For a given eps and N of trials
def run_experiment(m1, m2, m3, eps, N):

	bandits = [Bandit(m) for m in [m1,m2,m3]]

	#Array to save rewards of each iteration
	data = np.empty(N)

	for i in range(N):
		p = np.random.random()
		#With probability eps we choose a random bandit
		if p < eps:
			j = np.random.choice(3)
		else:#Else we choose the best bandit
			j = np.argmax([b.mean for b in bandits])
		#Get reward of chosen bandit and save it
		x = bandits[j].pull()
		bandits[j].update(x)

		#Keep track of rewards
		data[i] = x

	#Save cumulative reward
	#For each step [1,N], save cumulatitive rewards
	#This will show us, as time passes, how does our reward behave
	#As time increases we expect the cumulative_average to get closer to the best bandit, as we will learn that it is the best with time
	cumulative_average = np.cumsum(data)/(np.arange(N)+1)

	return cumulative_average

if __name__ == '__main__':
	#We run 3 experiments with different epsilons (our degree of willingness to explore more)
	#We expect lower epsilons to be better in the long run, but take longer in the beggining to find the best bandit
	c_1 = run_experiment(1,2,3,0.1,100000)
	c_05 = run_experiment(1.0, 2.0, 3.0, 0.05, 100000)
	c_01 = run_experiment(1.0, 2.0, 3.0, 0.01, 100000)

	plt.plot(c_1, label = 'ep = 0.1')
	plt.plot(c_05, label = 'ep = 0.05')
	plt.plot(c_01, label = 'ep = 0.01')
	plt.legend()
	plt.xlim(left=1)
	plt.xscale('log')
	plt.show()
