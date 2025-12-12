import sys
import random
import numpy as np
from project3 import ParticleFilter

# Function to get maze from file
#     Returns matrix with 1s for walls and 0 for walkable spaces
def load_maze(filename):
	with open(filename, 'r') as fp:
		lines = fp.readlines()
	maze = []
	for line in lines:
		walls = [1 if x == '#' else 0 for x in line.strip()]
		maze.append(walls)
	return maze

# Function to get an observation from the current robot position
#     Returns the distances (integers) to the closest walls ahead and behind the robot
#     Distance measurements are:
#     - exact 80% of the times
#     - shorter by 1 unit 10% of the times
#     - longer by 1 unit 10% of the times
def sense(maze, pos):
	i, j, d = pos	

	delta = {'up':(-1,0), 'down':(1,0), 'left':(0,-1), 'right':(0,1)}
	di, dj = delta[d]

	front = 0
	while maze[i+front*di][j+front*dj] == 0:
		front += 1

	back = 0
	while maze[i-back*di][j-back*dj] == 0:
		back += 1

	front = np.random.choice([front-1,front,front+1], 1, p=[0.1,0.8,0.1])
	back = np.random.choice([back-1,back,back+1], 1, p=[0.1,0.8,0.1])

	return front, back

# Function to move the robot randomly
#     Returns new position and action taken
#     Actions succeed 90% of the time. In the remaining 10%, nothing happens.
#     If action is 'move-forward' and the position in front of the robot is a wall, the action always fail.
def move(maze, pos):
	moves = ['rotate-left', 'rotate-right', 'move-forward']
	m = random.choice(moves)

	if random.random() > 0.9:
		return pos, m

	i, j, d = pos
	if m == 'move-forward':
		delta = {'up':(-1,0), 'down':(1,0), 'left':(0,-1), 'right':(0,1)}
		di, dj = delta[d]
		if maze[i+di][j+dj] == 0:
			i+=di
			j+=dj
	elif m == 'rotate-right':
		cw_order = ['up', 'right', 'down', 'left']
		k = cw_order.index(d)
		k = (k+1)%4
		d = cw_order[k]
	else:
		ccw_order = ['up', 'left', 'down', 'right']
		k = ccw_order.index(d)
		k = (k+1)%4
		d = ccw_order[k]

	return (i,j,d), m

maze = load_maze(sys.argv[1])

valid_pos = []
for i in range(len(maze)):
	for j in range(len(maze[0])):
		if maze[i][j] == 0:
			for k in ['up', 'down', 'left', 'right']:
				valid_pos.append((i,j,k))

position = random.choice(valid_pos)

# initialize particles with one observation
dfront, dback = sense(maze, position)
pf = ParticleFilter(maze, dfront, dback)

for i in range(30):
	# move randomly
	position, action = move(maze, position)
	pf.elapse(action)

	# get next observation
	dfront, dback = sense(maze, position)
	pf.observe(dfront, dback)

	# update particles
	pf.resample()

answer = pf.guess()
if answer == position:
	print('CORRECT')
else:
	print('INCORRECT')

