from enum import Enum, auto, unique
import pygame as pg
from pygame.math import Vector2
from vi import Agent, Simulation, util
from vi.config import Config, dataclass, deserialize, Window
import random
from typing import Optional
import polars as pl
import seaborn as sns

CHANCE_GENERATOR_MAX = 999

# rabbit Constants
RABBIT_COUNT = 25
REPRODUCTION_THRESHOLD_RABBIT = 995
RABBIT_CAPACITY = 250
RABBIT_GROWTH_RATE = 0.5
RABBIT_GROWTH_TIME = 50
RABBIT_ENERGY = 60

#Fox constants
FOX_COUNT = 10
REPRODUCTION_THRESHOLD_FOX = 998
FOX_ENERGY = 80 #between 0 and 100. Fox dies when 0 is reached.
ENERGY_GAIN_FOX = 50
ENERGY_LOSS_FOX = 0.4
ENERGY_MAX_FOX = 100
FOX_CAPACITY = 250
FOX_GROWTH_RATE = 0.5
FOX_GROWTH_TIME = 80

#Simulation hyperparams
MOVEMENT_SPEED = 1
RADIUS = 30
SEED = 1
DURATION = 120 * 60

#Plot constants
RESOLUTION = 200
Y_AXIS_MAX = 250


# Global vars
rabbit_count = RABBIT_COUNT
reproduction_timer_rabbit = 0
reproduction_amount_rabbit = 0
reproduction_amount_rabbit_max = 0
fox_count = FOX_COUNT
reproduction_timer_fox = 0
reproduction_amount_fox = 0
reproduction_amount_fox_max = 0

rabbit_numbered = 1

@deserialize
@dataclass
class FlockingConfig(Config):
    grass_grow_prob: float = 0.5

    movement_speed: float = 1
    delta_time: float = 3

    mass: int = 20


@unique
class States(Enum):
    WANDERING = auto()
    STILL = auto()


class Fox(Agent):
    config: FlockingConfig

    def on_spawn(self):
        self._state = States.WANDERING
        self.death_cause = "alive"
        self.energy = FOX_ENERGY
        self.hunger = 100 - FOX_ENERGY
        self.change_image(0)
        self.died = False
        #print(f"Foxes: {fox_count}, rabbits: {rabbit_count}")

    def update(self):

        rabbits = self.in_proximity_accuracy().without_distance().filter_kind(Rabbit)

        if rabbits is not None and self.hunger > 20:
            for obj_rabbit in rabbits:
                if not obj_rabbit.died:
                    obj_rabbit.die()
                    self.eat()
                    break

        global reproduction_timer_fox
        global reproduction_amount_fox
        global reproduction_amount_fox_max

        reproduction_timer_fox += 1
        if reproduction_timer_fox >= fox_count * FOX_GROWTH_TIME:
            reproduction_timer_fox = 0
            reproduction_amount_fox = max(int(FOX_GROWTH_RATE * fox_count * ((FOX_CAPACITY - fox_count) / FOX_CAPACITY) + 0.5), 1)
            reproduction_amount_fox_max = reproduction_amount_fox
            print(f"foxes: {fox_count} produce: {reproduction_amount_fox}")

        if reproduction_amount_fox > 0:
            reproduction_progress = 1 - reproduction_amount_fox / reproduction_amount_fox_max
            time_progress = reproduction_timer_fox / (fox_count * FOX_GROWTH_TIME)
            if reproduction_progress < time_progress:
                self.reproduction()
                reproduction_amount_fox -= 1
                print(f"produce: {reproduction_amount_fox}")

        self.lose_energy()

        if self.energy <= 0 and not self.died:
            self.die()

        self.save_data("kind", 0)

    def die(self):
        self.died = True
        self.kill()
        global fox_count
        fox_count -= 1
        #print(f"Foxes: {fox_count}, rabbits: {rabbit_count}")

    def eat(self):
        # Replenish energy
        # Assumes only one can be eaten even if more rabbits are in proximity
        self.energy = min(self.energy + ENERGY_GAIN_FOX, ENERGY_MAX_FOX)
        self.hunger = 100 - self.energy


    def lose_energy(self):
        # Decrease energy over time
        self.energy = max(self.energy - ENERGY_LOSS_FOX, 0)
        self.hunger = 100 - self.energy

    def reproduction(self):
        global fox_count
        self.reproduce()
        fox_count += 1


class Rabbit(Agent):
    config: FlockingConfig

    def on_spawn(self):
        self._state = States.WANDERING
        self.change_image(0)
        self.died = False
        self.energy = RABBIT_ENERGY
        #print(f"Foxes: {fox_count}, rabbits: {rabbit_count}")

    def update(self):
        global reproduction_timer_rabbit
        global reproduction_amount_rabbit
        global reproduction_amount_rabbit_max

       
        #  decrease in energy If the rabbit has no energy, it dies
        self.lose_energy()
        if self.energy <= 0 and not self.died:
            self.die()

       
        # Check if there is grass near
        grass = (self.in_proximity_accuracy()
                     .without_distance()
                     .filter_kind(Grass)
                     .first()
                )
        # If there is a grass, eat it and energy increases
        if grass is not None:
            grass.death_cause = "eaten"
            grass.kill()
            self.energy += 1

        reproduction_timer_rabbit += 1
        if reproduction_timer_rabbit >= rabbit_count * RABBIT_GROWTH_TIME:
            reproduction_timer_rabbit = 0
            reproduction_amount_rabbit = max(int(RABBIT_GROWTH_RATE * rabbit_count * ((RABBIT_CAPACITY - rabbit_count) / RABBIT_CAPACITY) + 0.5), 1)
            reproduction_amount_rabbit_max = reproduction_amount_rabbit
            print(f"rabbits: {rabbit_count} produce: {reproduction_amount_rabbit}")

        if reproduction_amount_rabbit > 0:
            reproduction_progress = 1 - reproduction_amount_rabbit / reproduction_amount_rabbit_max
            time_progress = reproduction_timer_rabbit / (rabbit_count * RABBIT_GROWTH_TIME)
            if reproduction_progress < time_progress:
                self.reproduction()
                reproduction_amount_rabbit -= 1
                print(f"produce: {reproduction_amount_rabbit}")

        self.save_data("kind", 1)

    def reproduction(self):
        global rabbit_count
        self.reproduce()
        rabbit_count += 1

    def die(self):
        global rabbit_count
        rabbit_count -= 1
        self.died = True
        self.kill()
        #print(f"Foxes: {fox_count}, rabbits: {rabbit_count}")

    def lose_energy(self):
        # Decrease energy over time
        self.energy = max(self.energy - ENERGY_LOSS_FOX, 0)
        self.hunger = 100 - self.energy

class Grass(Agent):
    config: FlockingConfig

    def on_spawn(self):
        self.change_image(0)
        
        # The grass grows in random places & it does not move
        self.death_cause = "alive"
        # The grass does not move
        self.freeze_movement()

    def update(self):
    
        # The grass grows back if it dies with given probability
        if util.probability(self.config.grass_grow_prob) and self.is_dead():
            self.reproduce()
        self.save_data("kind", 2)

class FlockingLive(Simulation):
    config: FlockingConfig

    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)

        self._background.fill((0, 0, 0))

(
    FlockingLive(
        FlockingConfig(
            image_rotation=True,
            movement_speed= MOVEMENT_SPEED,
            radius= RADIUS,
            seed= SEED,
            duration = DURATION
        )
    )

    .batch_spawn_agents(FOX_COUNT, Fox, images=["/Users/ania/Desktop/Collective_Intelligence_Group_Ocean/images/red.png"])
    .batch_spawn_agents(RABBIT_COUNT, Rabbit, images=["/Users/ania/Desktop/Collective_Intelligence_Group_Ocean/images/white.png"])
    .batch_spawn_agents(50, Grass, images=["/Users/ania/Desktop/Collective_Intelligence_Group_Ocean/images/grass2.png"])
    .run()
)






