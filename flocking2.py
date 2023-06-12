from enum import Enum, auto

import pygame as pg
from pygame.math import Vector2
from vi import Agent, Simulation
from vi.config import Config, dataclass, deserialize
import numpy as np

@deserialize
@dataclass
class FlockingConfig(Config):
    alignment_weight: float = 0.5
    cohesion_weight: float = 0.5
    separation_weight: float = 0.5

    delta_time: float = 3

    mass: int = 20

    def weights(self) -> tuple[float, float, float]:
        return (self.alignment_weight, self.cohesion_weight, self.separation_weight)


class Bird(Agent):
    config: FlockingConfig
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.max_velocity = 5.0 #set maximal velocity


    def change_position(self):
        # Pac-man-style teleport to the other end of the screen when trying to escape
        self.there_is_no_escape()
       
        #YOUR CODE HERE -----------
        #check teh neighbours in the radius
        neighbours = list(self.in_proximity_accuracy())
        n = len(neighbours)

         

        #if no neighbours then it wanders
        if n ==0:
            # Wander: simple movement behavior to start moving.
            current_velocity = self.move.length()  # the magnitude of the current move
            if current_velocity > 0:  # to avoid dividing by zero
                current_direction = (self.move + Vector2(np.random.uniform(-1,1), np.random.uniform(-1,1))).normalize()
                self.move = current_direction * self.max_velocity  # set move to the max velocity in the current direction
                self.pos += self.move

        #else (there are neighbours) so we need to use alignment, cohesion and separation
        else:
            #create the vectors multiplied by weights
            alignment = self.compute_alignment() * self.config.alignment_weight
            separation = self.compute_separation() * self.config.separation_weight
            cohesion = self.compute_cohesion() * self.config.cohesion_weight
            

            #calculating the total force
            f_total = ((alignment + separation + cohesion ) / self.config.mass)*self.config.delta_time

            # Adding force to movement
            self.move += f_total

            # Limiting to maximum velocity
            if self.move.length() > self.max_velocity:  # Using length() to get the velocity and compare to max_velocity
                self.move.normalize_ip()  # Normalizing move vector in-place to get vector length 1
                self.move.scale_to_length(self.max_velocity) # Scales vector to max_velocity value

            # Obstacle Avoidance
            obstacle_hit = pg.sprite.spritecollideany(self, self._obstacles, pg.sprite.collide_mask)  # type: ignore
            collision = bool(obstacle_hit)


            # Reverse direction when colliding with an obstacle.
            if collision:
                self.move.rotate_ip(90)


            # Check if the bird collides with an obstacle and "kill" it.
            # "kill" removes the sprite from all groups.
            for obstacle in self._obstacles:
                if pg.sprite.collide_rect(self, obstacle):
                    self.kill()


            # Update position, current position plus movement position value
            self.pos = self.pos + self.move 



    def compute_separation(self):
        separation = Vector2()
        neighbours = list(self.in_proximity_accuracy())
        n = len(neighbours)
        for neighbour, distance in neighbours:
            #sum of boid's position minus  neighbor boid’s position
            separation += self.pos - neighbour.pos
        #average of the separation value
        separation = separation/n
        return separation
    
    def compute_alignment(self):
        alignment = Vector2()
        neighbours = list(self.in_proximity_accuracy())
        n = len(neighbours)
        for neighbour, distance in neighbours:
            #sum of velocities of the boids in the neighbourhood
            alignment += neighbour.move
        #average of velocities - velocity of the boid
        alignment = alignment/n -  self.move
        return alignment
    
    def compute_cohesion(self):
        cohesion = Vector2()
        neighbours = list(self.in_proximity_accuracy())
        n = len(neighbours)
        for neighbour, distance in neighbours:
            #sum of position of neighboring boids
            cohesion= neighbour.pos
        #average of position of neighboring boids
        cohesion = cohesion/n
        #the cohesion force
        fc = cohesion - self.pos
        #calculating the cohesion
        cohesion = fc - self.move
        return cohesion
    def avoid_obstacle(self):
        obstacle = self.get_obstacle()
        if obstacle is not None:
            obstacle_pos = obstacle.position
            obstacle_radius = obstacle.radius

            to_obstacle = obstacle_pos - self.pos
            distance = to_obstacle.length()

            if distance < obstacle_radius + self.radius:
                desired_velocity = -to_obstacle.normalize() * self.max_velocity
                avoidance_force = (desired_velocity - self.move) / self.config.mass
                self.move += avoidance_force

    def get_obstacle(self):
        for entity in self.entities:
            if isinstance(entity, Obstacle):
                return entity
    

    


        #END CODE -----------------

class MovingObstacle(Obstacle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_velocity = 2.0 # max speed for obstacle
        self.move = Vector2(np.random.uniform(-1, 1), np.random.uniform(-1, 1)).normalize() * self.max_velocity

    def move_obstacle(self):
        self.pos += self.move

        # Similar to the Bird's movement, we will keep the obstacle within the screen
        self.there_is_no_escape()

class Selection(Enum):
    ALIGNMENT = auto()
    COHESION = auto()
    SEPARATION = auto()


class FlockingLive(Simulation):
    selection: Selection = Selection.ALIGNMENT
    config: FlockingConfig

    # ... other existing methods here ...

    def run(self):
        # Run the simulation.
        while self.running:
            # ... other existing game loop code here ...

            # Update obstacle positions
            for obstacle in self._obstacles:
                obstacle.move_obstacle()

    def handle_event(self, by: float):
        if self.selection == Selection.ALIGNMENT:
            self.config.alignment_weight += by
        elif self.selection == Selection.COHESION:
            self.config.cohesion_weight += by
        elif self.selection == Selection.SEPARATION:
            self.config.separation_weight += by

    def before_update(self):
        super().before_update()

        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_UP:
                    self.handle_event(by=0.1)
                elif event.key == pg.K_DOWN:
                    self.handle_event(by=-0.1)
                elif event.key == pg.K_1:
                    self.selection = Selection.ALIGNMENT
                elif event.key == pg.K_2:
                    self.selection = Selection.COHESION
                elif event.key == pg.K_3:
                    self.selection = Selection.SEPARATION

        a, c, s = self.config.weights()
        print(f"A: {a:.1f} - C: {c:.1f} - S: {s:.1f}")


(
    FlockingLive(
        FlockingConfig(
            image_rotation=True,
            movement_speed=1,
            radius=50,
            seed=1,
        )
    )
    .batch_spawn_agents(50, Bird, images=["images/bird.png"])
    .spawn_moving_obstacle(MovingObstacle, "images/triangle@50px.png", x=375, y=375)
    .run()
)

