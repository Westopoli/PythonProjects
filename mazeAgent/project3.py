import copy
import random

class ParticleFilter:
	def __init__(self, maze, front, back):
		self.maze = copy.deepcopy(maze)
		# self.pos = self.__get_positions_matching_obs(front, back)
		self.pos = self.__initialize_particles(front, back, n=30000)

	def __get_obs_for_position(self, pos):
		i, j, d = pos
		delta = {'up':(-1,0), 'down':(1,0), 'left':(0,-1), 'right':(0,1)}
		di, dj = delta[d]

		front = 0
		while self.maze[i+front*di][j+front*dj] == 0:
			front += 1

		back = 0
		while self.maze[i-back*di][j-back*dj] == 0:
			back += 1

		return front, back

	def __get_positions_matching_obs(self, front, back):
		all_pos = []
		for i in range(len(self.maze)):
			for j in range(len(self.maze[0])):
				if self.maze[i][j] == 0:
					for k in ['up', 'down', 'left', 'right']:
						mfront, mback = self.__get_obs_for_position((i,j,k))
						if mfront == front and mback == back:
							all_pos.append((i,j,k))
		return all_pos
	
	# def __initialize_particles(self, front, back, n=20000):
	# 	candidates = []
	# 	weights = []
		
	# 	# Consider wider range of sensor noise
	# 	for df in [front-1, front, front+1]:
	# 		for db in [back-1, back, back+1]:
	# 			matches = self.__get_positions_matching_obs(df, db)
	# 			for pos in matches:
	# 				candidates.append(pos)
	# 				# Weight by how likely we'd observe (front, back) from this position
	# 				# The position has true readings (df, db)
	# 				weight = self.__calculate_weight(front, back, df, db)
	# 				weights.append(weight)
		
	# 	# Remove duplicates and combine their weights
	# 	pos_to_weight = {}
	# 	for pos, w in zip(candidates, weights):
	# 		if pos in pos_to_weight:
	# 			pos_to_weight[pos] += w
	# 		else:
	# 			pos_to_weight[pos] = w
		
	# 	if not pos_to_weight:
	# 		# Fallback to all valid positions
	# 		for i in range(len(self.maze)):
	# 			for j in range(len(self.maze[0])):
	# 				if self.maze[i][j] == 0:
	# 					for d in ['up', 'down', 'left', 'right']:
	# 						pos_to_weight[(i, j, d)] = 1.0
		
	# 	candidates = list(pos_to_weight.keys())
	# 	weights = list(pos_to_weight.values())
		
	# 	# Sample particles weighted by initial observation likelihood
	# 	if sum(weights) > 0:
	# 		probs = [w/sum(weights) for w in weights]
	# 		return random.choices(candidates, weights=probs, k=min(n, len(candidates)))
	# 	else:
	# 		return random.choices(candidates, k=min(n, len(candidates)))

	def __initialize_particles(self, front, back, n=30000):
		candidates = []
		weights = []
		
		# Consider sensor noise range
		for df in [front-2, front-1, front, front+1, front+2]:
			for db in [back-2, back-1, back, back+1, back+2]:
				matches = self.__get_positions_matching_obs(df, db)
				for pos in matches:
					candidates.append(pos)
					weight = self.__calculate_weight(front, back, df, db)
					weights.append(weight)
		
		# Remove duplicates and combine weights
		pos_to_weight = {}
		for pos, w in zip(candidates, weights):
			if pos in pos_to_weight:
				pos_to_weight[pos] += w
			else:
				pos_to_weight[pos] = w
		
		if not pos_to_weight:
			# Fallback
			for i in range(len(self.maze)):
				for j in range(len(self.maze[0])):
					if self.maze[i][j] == 0:
						for d in ['up', 'down', 'left', 'right']:
							pos_to_weight[(i, j, d)] = 1.0
		
		candidates = list(pos_to_weight.keys())
		weights = list(pos_to_weight.values())
		
		if sum(weights) > 0:
			probs = [w/sum(weights) for w in weights]
			return random.choices(candidates, weights=probs, k=min(n, len(candidates)))
		else:
			return random.choices(candidates, k=min(n, len(candidates)))
	
	def __apply_action(self, pos, action):
		i, j, d = pos
		
		if action == 'move-forward':
			delta = {'up':(-1,0), 'down':(1,0), 'left':(0,-1), 'right':(0,1)}
			di, dj = delta[d]
			# Only move if next cell is walkable
			if self.maze[i+di][j+dj] == 0:
				return (i+di, j+dj, d)
			return pos  # Stay in place if wall ahead
			
		elif action == 'rotate-right':
			cw_order = ['up', 'right', 'down', 'left']
			k = cw_order.index(d)
			new_d = cw_order[(k+1) % 4]
			return (i, j, new_d)
			
		elif action == 'rotate-left':
			ccw_order = ['up', 'left', 'down', 'right']
			k = ccw_order.index(d)
			new_d = ccw_order[(k+1) % 4]
			return (i, j, new_d)
		
		return pos
	
	def __calculate_weight(self, obs_front, obs_back, particle_front, particle_back):
		# Probability model: P(observation | true_distance)
		# 80% exact, 10% off by -1, 10% off by +1
		
		def prob_sensor(observed, true):
			diff = observed - true
			if diff == 0:
				return 0.8
			elif diff == 1 or diff == -1:
				return 0.1
			else:
				return 0.001  # Very small for larger errors
		
		# Independent measurements: multiply probabilities
		weight = prob_sensor(obs_front, particle_front) * prob_sensor(obs_back, particle_back)
		return weight

	def elapse(self, action):
		new_pos = []
		for pos in self.pos:
			# 90% chance action succeeds, 10% stays same
			if random.random() < 0.9:
				new_pos.append(self.__apply_action(pos, action))
			else:
				new_pos.append(pos)
		self.pos = new_pos

	def observe(self, front, back):
		# Calculate weight for each particle
		self.weights = []
		for pos in self.pos:
			mfront, mback = self.__get_obs_for_position(pos)
			weight = self.__calculate_weight(front, back, mfront, mback)
			self.weights.append(weight)

	# def resample(self):
	# 	if sum(self.weights) == 0:
	# 		return
		
	# 	# Normalize weights
	# 	total = sum(self.weights)
	# 	probs = [w/total for w in self.weights]
		
	# 	# Low-variance resampling
	# 	n = len(self.pos)
	# 	new_pos = []
	# 	r = random.random() / n
	# 	c = probs[0]
	# 	i = 0
		
	# 	for m in range(n):
	# 		u = r + m / n
	# 		while u > c:
	# 			i += 1
	# 			if i >= len(probs):
	# 				i = len(probs) - 1
	# 				break
	# 			c += probs[i]
	# 		new_pos.append(self.pos[i])
		
	# 	self.pos = new_pos
	# 	self.weights = [1.0] * len(self.pos)

	def resample(self):
		if sum(self.weights) == 0 or sum(self.weights) < 1e-50:
			# Particle deprivation - reinitialize with random particles
			print("Warning: particle deprivation, reinitializing")
			valid_positions = []
			for i in range(len(self.maze)):
				for j in range(len(self.maze[0])):
					if self.maze[i][j] == 0:
						for d in ['up', 'down', 'left', 'right']:
							valid_positions.append((i, j, d))
			self.pos = random.choices(valid_positions, k=len(self.pos))
			self.weights = [1.0] * len(self.pos)
			return
		
		total = sum(self.weights)
		probs = [w/total for w in self.weights]
		
		# Low-variance resampling
		n = len(self.pos)
		new_pos = []
		r = random.random() / n
		c = probs[0]
		i = 0
		
		for m in range(n):
			u = r + m / n
			while u > c:
				i += 1
				if i >= len(probs):
					i = len(probs) - 1
					break
				c += probs[i]
			new_pos.append(self.pos[i])
		
		# Add small percentage of random particles (2-5%)
		n_random = max(1, int(0.03 * n))
		valid_positions = []
		for i in range(len(self.maze)):
			for j in range(len(self.maze[0])):
				if self.maze[i][j] == 0:
					for d in ['up', 'down', 'left', 'right']:
						valid_positions.append((i, j, d))
		
		# Replace last n_random particles with random ones
		for idx in range(n_random):
			new_pos[-(idx+1)] = random.choice(valid_positions)
		
		self.pos = new_pos
		self.weights = [1.0] * len(self.pos)

	# def guess(self):
	# 	if self.pos:
	# 		# Return the most frequent particle (mode)
	# 		from collections import Counter
	# 		counts = Counter(self.pos)
	# 		return counts.most_common(1)[0][0]
	# 	else:
	# 		for i in range(len(self.maze)):
	# 			for j in range(len(self.maze[0])):
	# 				if self.maze[i][j] == 0:
	# 					return (i, j, 'up')
	# 		return (1, 1, 'up')

	def guess(self):
		if not self.pos:
			for i in range(len(self.maze)):
				for j in range(len(self.maze[0])):
					if self.maze[i][j] == 0:
						return (i, j, 'up')
			return (1, 1, 'up')
		
		# Return most common particle (mode)
		from collections import Counter
		counts = Counter(self.pos)
		
		# If there's a clear winner (>10% of particles), return it
		most_common = counts.most_common(1)[0]
		if most_common[1] > len(self.pos) * 0.1:
			return most_common[0]
		
		# Otherwise, consider top candidates
		# Return the one with best average weight
		top_candidates = counts.most_common(5)
		return top_candidates[0][0]
