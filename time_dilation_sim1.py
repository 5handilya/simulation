#simulation of time dilation

import numpy as np
np.set_printoptions(precision=15)
import pygame
import math
from pygame import Vector2
import sys

class Body:
    def __init__(self, mass, position, velocity, color):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.dilated_position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.array([0.0, 0.0, 0.0])
        self.color = color
        self.time = 0

class GravitationalSystem:
    #units: m,kg,s
    G = 6.6743e-11
    c = 299752458
    def __init__(self):
        #MASS
        mass_sun = 1.989e30
        mass_mercury = 0.330e24 
        mass_earth = 5.9722e24 

        pdist_mercury = 4.579e10
        pdist_earth = 1.4709807e11
        
        pvel_mercury = 5.897e4
        pvel_earth = 3.029e4

        zmax_mercury = 1.05e10
        zmax_earth = 0

        self.bodies = [
            Body(mass_sun, [0,0,0], [0,0,0], (255,255,0)),
            
            Body(mass_mercury, [pdist_mercury,0,zmax_mercury], [0,pvel_mercury,0], (10,255,0)),
            
            Body(mass_earth, [pdist_earth,0,zmax_earth], [0,pvel_earth,0], (100,150,255))
        ]

        #SIM PARAMS
        #time step in s
        self.dt = 10000
        #pixels per 1e6 km
        self.scale = 4e8

    def calculate_acceleration(self, bodies):
        accelerations = [np.array([0.0, 0.0, 0.0]) for _ in bodies]
        for i in range (len(bodies)):
            for j in range(i+1, len(bodies)):
                r = bodies[j].position - bodies[i].position
                r_mag = np.linalg.norm(r)
                force_mag = self.G * bodies[i].mass * bodies[j].mass / (r_mag ** 2)
                force = force_mag * r / r_mag 
                accelerations[i] += force / bodies[i].mass
                accelerations[j] -= force / bodies[j].mass
        return accelerations

    def verlet_step(self):
        init_accel = self.calculate_acceleration(self.bodies)
        for i, body in enumerate(self.bodies):
            lorentz_factor = 1
            if i != 0:
                vel_mag = np.linalg.norm(body.velocity)
                lorentz_factor = 1.0 / np.sqrt((1.0 - (vel_mag / self.c) ** 2), dtype=np.float128)
                body.time += self.dt / lorentz_factor
            else:
                body.time += self.dt
            body.position += body.velocity * self.dt + 0.5 * init_accel[i] * self.dt **2
            body.dilated_position += body.velocity * self.dt/lorentz_factor + 0.5 * init_accel[i] * self.dt/lorentz_factor **2
            
        new_accel = self.calculate_acceleration(self.bodies)
        for i,body in enumerate (self.bodies):
            body.velocity += 0.5 * (init_accel[i] + new_accel[i]) * self.dt


class Simulator:
    def __init__(self, width=1220, height=1080):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("time dilation demo")
        self.system = GravitationalSystem()
        self.clock = pygame.time.Clock()
        
    def world_to_screen(self, position):
        screen_x = position[0] / self.system.scale + self.width // 2
        screen_y = position[1]/ self.system.scale + self.height // 2
        return int(screen_x), int(screen_y)

    def render_text(self):
        font = pygame.font.Font(None, 42)
        sun, mercury, earth = self.system.bodies[0], self.system.bodies[1], self.system.bodies[2]
        sun_time_text = f"observer at rest (s): ({sun.time:.1e})"
        mercury_time_text = f"observer - mercury(s): ({mercury.time - sun.time})"
        earth_time_text = f"observer - earth(s): ({earth.time - sun.time})"
        sun_surface= font.render(sun_time_text, True, (255, 255, 255))
        mercury_surface = font.render(mercury_time_text, True, (255, 255, 255))
        earth_surface = font.render(earth_time_text, True, (255, 255, 255))
        self.screen.blit(sun_surface, (self.width - 700, 50))
        self.screen.blit(mercury_surface, (self.width - 700, 80))
        self.screen.blit(earth_surface, (self.width - 700, 110))

    def draw_clock(self, x, y, time, color):
        # Clock parameters
        radius = 50
        center = (x + radius, y + radius)
        
        # Draw clock face
        pygame.draw.circle(self.screen, (255, 255, 255), center, radius, 2)
        
        # Calculate hand angle (one rotation = 0.01 second)
        # Convert time to rotations, then to radians
        angle = (time % 0.01) * (2 * math.pi / 0.01)
        
        # Calculate hand endpoint
        hand_length = radius - 10
        end_x = center[0] + hand_length * math.sin(angle)
        end_y = center[1] - hand_length * math.cos(angle)
        
        # Draw hand
        pygame.draw.line(self.screen, color, center, (end_x, end_y), 2)

    def render_clocks(self):
        # Draw three clocks vertically on the left side
        sun, mercury, earth = self.system.bodies[0], self.system.bodies[1], self.system.bodies[2]
        
        # Draw clocks at different vertical positions
        self.draw_clock(50, 50, 0, sun.color)      # Sun clock
        self.draw_clock(50, 170, mercury.time, mercury.color)  # Mercury clock
        self.draw_clock(50, 290, earth.time, earth.color)    # Earth clock
        
        # Add labels
        font = pygame.font.Font(None, 32)
        labels = ["observer at rest", "rest - mercury", "rest - earth", "(all clocks: 360deg = 0.1s)"]
        for i, label in enumerate(labels):
            text = font.render(label, True, (255, 255, 255))
            self.screen.blit(text, (170, 95 + i * 120))
    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.system.verlet_step()
            self.screen.fill((0, 0, 0))
            if self.system.bodies[0].time > 7000000:
                self.system.dt = 100000
            if self.system.bodies[0].time > 400000000:
                self.system.dt = 10000
            
            for i,body in enumerate(self.system.bodies):
                pos = self.world_to_screen(body.position)
                radius = max(2, int(np.log2(body.mass) - 20)/6)
                #if body == self.system.bodies[0]:
                #    radius = 1
                if i != 0:
                    pygame.draw.circle(self.screen, (200,200,200), self.world_to_screen(body.dilated_position), radius)
                pygame.draw.circle(self.screen, body.color, pos, radius)
            self.render_text() 
            self.render_clocks()
            pygame.display.flip()
            self.clock.tick(60000)

if __name__ == "__main__":
    sim = Simulator()
    sim.run()
