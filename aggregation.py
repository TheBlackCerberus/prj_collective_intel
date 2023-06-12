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
        n = len(neighbours)

        if self.state == State.WANDERING:
            if n > 0 and np.random.uniform() < self.config.Pjoin:
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
        if self.state in [State.WANDERING, State.JOINING, State.LEAVING]:
            # Implement movement logic here
            pass

class AggregationSimulation(Simulation):
    config: AggregationConfig

    def run(self):
        while self.running:
            for agent in self._agents:
                agent.change_state()
                agent.move()
            # Implement rendering logic here