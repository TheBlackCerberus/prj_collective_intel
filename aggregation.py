from enum import Enum
import numpy as np
from vi import Agent, Simulation
from vi.config import Config, dataclass, deserialize

@deserialize
@dataclass
class AggregationConfig(Config):
    Pjoin: float = 0.5
    Pleave: float = 0.5
    Tjoin: int = 10
    Tleave: int = 10

class State(Enum):
    WANDERING = 1
    JOINING = 2
    STILL = 3
    LEAVING = 4

class Cockroach(Agent):
    config: AggregationConfig
    state: State
    timer: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = State.WANDERING
        self.timer = 0

    def change_state(self):
        neighbours = list(self.in_proximity_accuracy())
        num_neighbours = len(neighbours)

        if self.state == State.WANDERING:
            if num_neighbours > 0 and np.random.uniform() < self.config.Pjoin:
                self.state = State.JOINING
                self.timer = self.config.Tjoin
        elif self.state == State.JOINING:
            self.timer -= 1
            if self.timer <= 0:
                self.state = State.STILL
        elif self.state == State.STILL:
            if np.random.uniform() < self.config.Pleave:
                self.state = State.LEAVING
                self.timer = self.config.Tleave
        elif self.state == State.LEAVING:
            self.timer -= 1
            if self.timer <= 0:
                self.state = State.WANDERING

    def move(self):
        if self.state in [State.WANDERING, State.JOINING, State.LEAVING, State.STILL]:
            if self.state == State.WANDERING:
                # Random movement
                current_direction = (self.move + Vector2(np.random.uniform(-1,1), np.random.uniform(-1,1))).normalize()
                self.move = current_direction 
                # self.pos += self.move
            if self.state == State.JOINING:
                self.move = self.move + self.move_join
            if self.state == State.LEAVING:
                self.move = ..... 
            self.pos += self.move
   
    def state_wander(self):
        self.state = State.WANDERING
    
    def state_join(self, num_neighbors):
        # Probability of joining based on the number of neighbors
        p_join = num_neighbors * 0.1  # Adjust the coefficient according to your requirements
        if random.random() < p_join:
            self.state = State.JOINING
    
    def move_join(self):
        join = Vector2()
        neighbours = list(self.in_proximity_accuracy())
        n = len(neighbours)
        for neighbour, distance in neighbours:
            #sum of position of neighboring boids
            join += neighbour.pos
        #the join force
        fj = join - self.pos
        #calculating the cohesion
        join = fj - self.move
        return join
        

    def state_still(self, joining_timer):
        # Transition to Still state after a certain joining timer
        self.timer += 1
        if self.timer >= joining_timer:
            self.state = State.STILL
            self.timer = 0
    
    def move_still(self):
        self.pos 


    def state_leave(self, num_neighbors):
        # Probability of leaving based on the number of neighbors
        p_leave = num_neighbors * 0.1  # Adjust the coefficient according to your requirements
        if random.random() < p_leave:
            self.state = State.LEAVING
    
    def move_leave(self):
        self.pos = (self.move + Vector2(np.random.uniform(-1,1), np.random.uniform(-1,1))).normalize() #just move in a random  place


class AggregationSimulation(Simulation):
    config: AggregationConfig

    def run(self):
        while self.running:
            for agent in self._agents:
                agent.change_state()
                agent.move()
            # Implement rendering logic here