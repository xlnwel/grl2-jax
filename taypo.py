import numpy as np


def generate_trajectories(mdp, mu, num_simulations, T):
	"""
	Generate trajectories from the MDP using policy mu

	Args:
		mdp: the mdp object
		mu: the policy to be executed
		num_simulations: num of trajectories
		T: truncated horizon of the trajectory
	Returns:
		A list of trajectories
	"""
	trajs = []
	na = mu.shape[1]
	for i in range(num_simulations):
		rsum = []
		states = []
		actions = []
		s = mdp.reset()
		for t in range(T):
			a = np.random.choice(np.arange(na), p=mu[s])
			s_next, r, _, _ = mdp.step(a)
			rsum.append(r)
			actions.append(a)
			states.append(s)
			s = s_next
		states.append(s_next)
		trajs.append({'states': states, 'actions': actions, 'rewards': rsum})
	return trajs


if __name__ == '__main__':
	# --------------------
	# Adjust hyper-parameters here
	# --------------------
	num_simulations = 1000  # num of trajectories used for averaging estimates
	epsilon = 0.8  # off-policyness
	independent_trials = 10  # num of independent trials
	ns, na = 20, 5  # state and action space dimension
	gamma = 0.8  # discount factor
	T = 20  # trajectory length
	density = 0.001  # parameter for generating MDP
	rho = np.inf  # truncation hyper-parameter -- default to no truncation
	c = np.inf  # truncation hyper-parameter -- default to no truncation

	# --------------------
	# Create directory to save data
	# --------------------
	directory = 'hessian_and_gradient_offpolicy_{}_eps{}'.format(num_simulations, epsilon)
	if not os.path.exists(directory):
		os.makedirs(directory)

	# --------------------
	# Create list to hold computed data
	# --------------------
	nocritic_grad = []
	nocritic_hessian = []
	truncated_grad = []
	truncated_hessian = []
	firstorder_grad = []
	firstorder_hessian = []
	loadeddice_grad = []
	loadeddice_hessian = []
	secondorder_grad = []
	secondorder_hessian = []
