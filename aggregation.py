from enum import Enum
import numpy as np
from vi import Agent, Simulation
from vi.config import Config, dataclass, deserialize

@deserialize
@dataclass
class AggregationConfig(Config):
    Pjoin: float = 0.5
    Pleave: float = 0.5
    t_join: int = 10
    tINING = 2
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
        if 

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
        
            

class AggregationSimulation(Simulation):
    config: AggregationConfig

    def run(self):
        while self.running:
            for agent in self._agents:
                agent.change_state()
                agent.move()
            


(
    AggregationSimulation(config=
        AggregationConfig(
            Pjoin=0.5,
            Pleave=0.5,
            Tjoin=10,
            Tleave=10,
            movement_speed=10,
        )
        .batch_spawn_agents(50, Cockroach, images=["red.png"])
        .run()
    )
)