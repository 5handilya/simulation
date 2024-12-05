#2 body 2D gravitation simulator with verlet integration
import numpy as np
import pygame
from pygame import Vector2
import sys

class Body:
    def __init__(self, mass, position, velocity, color):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.array([0.0, 0.0])
        self.color = color

class GravitationalSystem:
    G = 6.67428e-11 
    def __init__(self):
        au = 1.496e11
        perihelion = 1.471e11
        sunmass = 1.989e30
        earthmass = 5.9722e24
        p_earthvel = 3e4 #velocity at perihelion = 30,000m/s
        #e.g. two bodies - earth and sun
        self.bodies = [
            #Body(2.0e30, [0, 0], [0, 0], (255, 255, 0)),  # static sun
            Body(sunmass, [0, 0], [0,0], (255, 255, 0)),  # sun
            Body(earthmass, [perihelion, 0], [0, p_earthvel], (0, 255, 255))  # planet 
            #Body(1.0e30, [1.5e11, 0], [0, 15000], (0, 255, 255))  # sun2
        ]
        # SIMULATION PARAMS
        self.dt = 360000  # time step (10 hour)
        self.scale = 1e9  # pixels per billion m
        
    def calculate_acceleration(self, body1, body2):
        r = body2.position - body1.position
        r_mag = np.linalg.norm(r)
        
        force_mag = self.G * body1.mass * body2.mass / (r_mag ** 2)
        force = force_mag * r / r_mag
        
        acc1 = force / body1.mass
        acc2 = -force / body2.mass
        
        return acc1, acc2
    
    def verlet_step(self):
        acc1, acc2 = self.calculate_acceleration(self.bodies[0], self.bodies[1])
        self.bodies[0].acceleration = acc1
        self.bodies[1].acceleration = acc2
        
        for body in self.bodies:
            body.position += body.velocity * self.dt + 0.5 * body.acceleration * self.dt**2
            
            old_acc = body.acceleration.copy()
            
            acc1, acc2 = self.calculate_acceleration(self.bodies[0], self.bodies[1])
            self.bodies[0].acceleration = acc1
            self.bodies[1].acceleration = acc2
            
            body.velocity += 0.5 * (old_acc + body.acceleration) * self.dt

class Simulator:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Two-Body Gravitational Simulation")
        
        self.system = GravitationalSystem()
        self.clock = pygame.time.Clock()
        
    def world_to_screen(self, position):
        # Convert world coordinates to screen coordinates
        screen_x = position[0] / self.system.scale + self.width // 2
        screen_y = position[1] / self.system.scale + self.height // 2
        return int(screen_x), int(screen_y)
    
    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            # Physics update
            self.system.verlet_step()
            
            # Drawing
            self.screen.fill((0, 0, 0))
            
            # Draw bodies
            for body in self.system.bodies:
                pos = self.world_to_screen(body.position)
                # Size based on mass (logarithmic scale)
                radius = int(np.log10(body.mass) - 20)
                pygame.draw.circle(self.screen, body.color, pos, radius)
            
            pygame.display.flip()
            self.clock.tick(60)

if __name__ == "__main__":
    sim = Simulator()
    sim.run()
