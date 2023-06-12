import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class CockroachAgent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.state = "Wandering"
        self.timer = 0

    def wander(self):
        # Random movement within the environment boundaries
        self.x += random.uniform(-1, 1)
        self.y += random.uniform(-1, 1)

    def join(self, num_neighbors):
        # Probability of joining based on the number of neighbors
        p_join = num_neighbors * 0.1  # Adjust the coefficient according to your requirements
        if random.random() < p_join:
            self.state = "Joining"

    def still(self, joining_timer):
        # Transition to Still state after a certain joining timer
        self.timer += 1
        if self.timer >= joining_timer:
            self.state = "Still"
            self.timer = 0

    def leave(self, num_neighbors):
        # Probability of leaving based on the number of neighbors
        p_leave = num_neighbors * 0.1  # Adjust the coefficient according to your requirements
        if random.random() < p_leave:
            self.state = "Leaving"

    def update(self, num_neighbors, joining_timer, target):
        if self.state == "Wandering":
            self.wander()
            self.join(num_neighbors)
        elif self.state == "Joining":
            self.join(num_neighbors)
            self.still(joining_timer)
        elif self.state == "Still":
            self.leave(num_neighbors)
        elif self.state == "Leaving":
            self.leave(num_neighbors)
            self.state = "Wandering"

        # Move towards the target circle
        target_x, target_y = target
        dx = target_x - self.x
        dy = target_y - self.y
        distance = (dx ** 2 + dy ** 2) ** 0.5
        if distance != 0:
            self.x += dx / distance
            self.y += dy / distance



# Simulation settings
num_agents = 20
num_neighbors = 5  # Example number of neighbors for testing
joining_timer = 10  # Example joining timer for testing
num_iterations = 1000
target = (2, 5)  # Coordinates of the target circle


# Initialize agents
agents = [CockroachAgent(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(num_agents)]

# Initialize plot
fig, ax = plt.subplots()

# Create target circle
target_circle = plt.Circle(target, radius=1.5, edgecolor='red', facecolor='none')
ax.add_patch(target_circle)


# Initialize empty scatter plot
scatter = ax.scatter([], [], marker='o', color='blue')

# Update function for animation
def update_plot(i):
    # Update agent positions and states
    for agent in agents:
        agent.update(num_neighbors, joining_timer, target)
    
    # Update scatter plot data
    scatter.set_offsets([(agent.x, agent.y) for agent in agents])
    
    # Set plot limits
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    
    # Set plot title and labels
    ax.set_title(f"Iteration {i}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

# Animate the simulation
ani = animation.FuncAnimation(fig, update_plot, frames=num_iterations, interval=200)
# Show the plot
plt.show()
