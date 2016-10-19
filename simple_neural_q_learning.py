import gym
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

env = gym.make('FrozenLake-v0')

# Agent can be in one of 16 states.
states = np.identity(16)

x = tf.placeholder(shape=[1, 16], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16, 4], 0, 0.1))

# Estimated Q values for each action.
y = tf.matmul(x, W)

# Observed Q values (well only one action is observed, the rest remain equal to 'y').
y_ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(y_ - y))
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

discount = 0.99
epsilon = 0.5

epsilon_history = []
reward_history = []

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())

	for i in range(5000):
		s = env.reset()
		game_over = False

		while not game_over:
			estimated_Qs = sess.run(y, {x: states[s:s + 1]})
			action = np.argmax(estimated_Qs, 1)[0]

			# Occasionally try a random action (explore).
			if np.random.rand(1) < epsilon:
				action = env.action_space.sample()

			# Perform the action and observe its actual Q value.
			next_s, reward, game_over, _ = env.step(action)
			observed_Q = reward + discount * np.max(sess.run(y, {x: states[next_s:next_s + 1]}))
			
			# Measure error of initial estimation and learn from it.
			estimated_Qs[0, action] = observed_Q
			sess.run(train_step, {x: states[s:s + 1], y_: estimated_Qs})

			s = next_s

		epsilon_history.append(epsilon)
		reward_history.append(reward)
		
		print('Episode: {:d}  Reward: {:g}  Epsilon: {:g}'.format(i + 1, reward, epsilon))

		epsilon *= 0.999

	# Test the agent.
	env.monitor.start('/tmp/simple_neural_qlearning_results')
	total_reward = 0

	for _ in range(100):
		s = env.reset()
		game_over = False

		while not game_over:
			Qs = sess.run(y, feed_dict={x: states[s:s + 1]})
			action = np.argmax(Qs, 1)[0]
			s, reward, game_over, _ = env.step(action)
			total_reward += reward

	env.monitor.close()
	print('Average Reward:', total_reward / 100)

plt.subplot(211)
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.plot(reward_history)

plt.subplot(212)
plt.ylabel('Explore / Exploit')
plt.xlabel('Episode')
plt.plot(epsilon_history)

plt.tight_layout()
plt.show()
