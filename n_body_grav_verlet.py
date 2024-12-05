#n body grav simulation using verlet integration
#visualized in pygame
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
        self.trail = []
class GravitationalSystem:
    #units: m, kg, s
    G = 6.67430e-11  # gravitational constant
    def __init__(self, num_bodies):
        au = 1.496e11
        perihelion = 1.471e11
        sunmass = 1.989e30
        earthmass = 5.9722e24
        p_earthvel = 3e4 #velocity at perihelion = 30,000m/s
        self.bodies = [
                Body(sunmass, [0, 0], [0, 0], (255, 255, 255)), 
                Body(earthmass, [perihelion, 0], [0, p_earthvel], (255, 0, 0)), 
                #Body(grav, [1.5e9, 1.5e9], [-vel, 0], (0, 0, 255)),
                #Body(grav, [1.5e9, 1.5e9], [-vel, 0], (0, 0, 255)),
                #Body(grav, [1.5e8, 1.5e8], [-vel, 0], (0, 0, 255)),
                #Body(grav*1e1, [-1e11, -1e11], [bigvel, bigvel], (255, 0, 255)),
                #Body(grav*1e1, [-1e11, 1e11], [bigvel, -bigvel], (128, 0, 255)),
                #Body(grav*1e1, [1e11, -1e11], [-bigvel, bigvel], (128, 0, 128)),
                #Body(grav*1e1, [1e11, 1e11], [-bigvel, -bigvel], (128, 128, 255)),
                #fast traveler blackhole
                #Body(grav*1e1, [-1e13, 7.5e9], [vel*1e2, 0], (128, 128, 255)),
                #Body(grav, [0, 1.5e9], [0, -vel], (0, 255, 0)),
                #Body(grav, [0, 0], [vel, 0], (255, 255, 0))
        ]
        # SIM PARAMS 
        self.dt = 3600 # time step in seconds
        self.scale = 1e9 # pixels per million km 
    def calculate_acceleration(self, bodies):
        accelerations = [np.array([0.0, 0.0]) for _ in bodies]
        for i in range(len(bodies)):
            for j in range(i+1, len(bodies)):
                r = bodies[j].position - bodies[i].position
                r_mag = np.linalg.norm(r)
                force_mag = self.G * bodies[i].mass * bodies[j].mass / (r_mag ** 2)
                force = force_mag * r / r_mag
                accelerations[i] += force / bodies[i].mass
                accelerations[j] -= force / bodies[j].mass
        return accelerations
    
    def verlet_step(self):
        old_accelerations = self.calculate_acceleration(self.bodies)
        for i, body in enumerate(self.bodies):
            body.position += body.velocity * self.dt + 0.5 * old_accelerations[i] * self.dt**2

        accelerations = self.calculate_acceleration(self.bodies)
        for i, body in enumerate(self.bodies):
            body.velocity += 0.5 * (old_accelerations[i] + accelerations[i]) * self.dt
            body.trail.append(body.position.copy())
            if len(body.trail) > 100:
                body.trail.pop(0)

class Simulator:
    def __init__(self, width=1920, height=1080):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("N-Body Gravitational Simulation")
        self.system = GravitationalSystem(2)
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
                # Draw trails
                for trail_pos in body.trail:
                    screen_pos = self.world_to_screen(trail_pos)
                    pygame.draw.circle(self.screen, body.color, screen_pos, 1)
                
                # Draw bodies
                pos = self.world_to_screen(body.position)
                # Size based on mass (logarithmic scale)
                radius = int(np.log10(body.mass) - 20)
                pygame.draw.circle(self.screen, body.color, pos, radius)
            
            pygame.display.flip()
            self.clock.tick(60000)

if __name__ == "__main__":
    sim = Simulator()
    sim.run()
