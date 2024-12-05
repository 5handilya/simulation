#n body grav simulation using verlet integration
#visualized in pygame
import numpy as np
import pygame
import math
from pygame import Vector2
import sys
class Body:
    def __init__(self, mass, position, velocity, color):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.array([0.0, 0.0, 0.0])
        self.color = color
        self.trail = []

class GravitationalSystem:
    #units: m, kg, s
    G = 6.67430e-11  # gravitational constant
    earth_sun_max_distance = 0
    def __init__(self, num_bodies):
        
        #MASS
        mass_sun = 1.989e30
        mass_circle = 1e24
        mass_mercury = 0.330e24
        mass_venus = 4.87e24
        mass_earth = 5.9722e24
        mass_moon = 7.346e22
        mass_mars = 0.64169e24
        mass_phobos = 1.0659e16 
        mass_deimos = 1.51e15 
        mass_jupiter = 1.89813e27
        mass_europa = 4.8e22
        mass_ganymede = 1.4819e23
        mass_io = 8.931938e22
        mass_saturn = 5.6832e24
        mass_uranus = 86.811e24
        mass_neptune = 1.02409e26

        #VELOCITY_AT_PERIHELION
        pvel_circle = 2.978e4
        pvel_mercury = 5.897e4
        pvel_venus = 3.526e4
        pvel_earth = 3.029e4
        pvel_mars = 2.65e4
        pvel_phobos = pvel_mars + 2.138e3
        pvel_deimos = pvel_mars + 1.3513e3
        pvel_jupiter = 1.372e4
        pvel_saturn = 1.014e4
        pvel_uranus = 7.13e3
        pvel_neptune = 5.47e3
            #moons
        pvel_moon = pvel_earth + 1082
        pvel_europa = pvel_jupiter + 1.37e4
        pvel_ganymede = pvel_jupiter + 1.088e4
        pvel_io = pvel_jupiter + 1.4334e4

        #PERIHELION 
        pdist_circle = 1.4709807e11
        pdist_mercury = 4.579e10
        pdist_venus = 1.0748e11
        pdist_earth = 1.4709807e11
        pdist_moon= pdist_earth + 3.633e8
        pdist_mars = 2.0665e11
        pdist_phobos = pdist_mars + 6e3
        pdist_deimos = pdist_mars + 2.3458e27
        pdist_jupiter = 7.40595e11
        pdist_europa = pdist_jupiter + 6.71e8 
        pdist_ganymede = pdist_jupiter + 1.0692e9
        pdist_io = pdist_jupiter + 4.2e8
        pdist_saturn = 1.357554e12
        pdist_uranus = 2.732696e12
        pdist_neptune = 4.471050e12

        #z-max
        zmax_mercury = 1.05e10
        zmax_earth = 42
        zmax_venus = 7.5e9
        zmax_mars = 4.5e9
        zmax_jupiter = 39
        zmax_saturn=6e9
        zmax_uranus=3e9
        zmax_neptune=4.5e9

        self.bodies = [
                Body(mass_sun, [0, 0, 0], [0, 0, 0], (255, 255, 0)), 
                Body(mass_earth, [pdist_earth, 0 , zmax_earth], [0, pvel_earth, 0], (0, 150, 245)), 
                Body(mass_moon, [pdist_moon, 0, zmax_earth], [0, pvel_moon, 0], (255, 255, 255)), 
                Body(mass_mercury, [pdist_mercury, 0, zmax_mercury], [0, pvel_mercury, 0], (255, 0, 128)), 
                Body(mass_venus, [pdist_venus, 0, zmax_venus], [0, pvel_venus, 0], (255, 128, 129)), 
                Body(mass_mars, [pdist_mars, 0, zmax_mars], [0, pvel_mars, 0], (255, 0, 0)), 
                Body(mass_phobos, [pdist_phobos, 0, zmax_mars], [0, pvel_phobos, 0], (255, 255, 255)), 
                Body(mass_deimos, [pdist_deimos, 0, zmax_mars], [0, pvel_deimos, 0], (255, 255, 255)), 
                Body(mass_jupiter, [pdist_jupiter, 0, zmax_jupiter], [0, pvel_jupiter, 0], (255, 128, 0)), 
                Body(mass_europa, [pdist_europa, 0, zmax_jupiter], [0, pvel_europa, 0], (255, 255, 255)), 
                Body(mass_ganymede, [pdist_ganymede, 0, zmax_jupiter], [0, pvel_ganymede, 0], (128, 128, 128)), 
                Body(mass_io, [pdist_io, 0,0], [0, pvel_io, zmax_jupiter], (255, 128, 244)), 
                Body(mass_saturn, [pdist_saturn, 0, zmax_saturn], [0, pvel_saturn, 0], (100,100, 0)), 
                Body(mass_uranus, [pdist_uranus, 0, zmax_uranus], [0, pvel_uranus, 0], (100, 100, 100)), 
                Body(mass_neptune, [pdist_neptune, 0, zmax_neptune], [0, pvel_neptune, 0], (0, 10, 255)), 
        ]
        # SIM PARAMS 
        # time step in seconds
        self.dt = 6000
        # pixels per million km
        self.scale = 1.5e10
    def calculate_acceleration(self, bodies):
        accelerations = [np.array([0.0, 0.0, 0.0]) for _ in bodies]
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
            if len(body.trail) > 1:
                body.trail.pop(0)
        

class Simulator:
    def __init__(self, width=800, height=800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("N-Body Gravitational Simulation")
        self.system = GravitationalSystem(15)
        self.clock = pygame.time.Clock()
        
    def world_to_screen(self, position):
        # Convert world coordinates to screen coordinates
        screen_x = position[0] / self.system.scale + self.width // 2
        screen_y = position[1] / self.system.scale + self.height // 2
        return int(screen_x), int(screen_y)

    def render_text(self):
        # Initialize font
        font = pygame.font.Font(None, 24)
        
        # Get the sun and earth bodies
        sun, earth = self.system.bodies[0], self.system.bodies[1]
        
        # Calculate distance
        distance = np.linalg.norm(earth.position - sun.position)
        
        # Calculate velocity magnitude
        velocity_mag = np.linalg.norm(earth.velocity)
        
        #Render xyz text
        position_text = f"Earth position: ({earth.position[0]:.1}, {earth.position[1]:.1}, {earth.position[2]:.1})"
        position_surface= font.render(position_text, True, (255, 255, 255))

        # Render distance text
        distance_text = f"Earth-sun distance: {distance/1e9:.2f} million km"
        distance_surface = font.render(distance_text, True, (255, 255, 255))
        
        # Render velocity text
        velocity_text = f"Earth orbital velocity: {velocity_mag/1000:.2f} km/s"
        velocity_surface = font.render(velocity_text, True, (255, 255, 255))
        
        # Position texts in top right corner
        self.screen.blit(distance_surface, (self.width - 300, 10))
        self.screen.blit(velocity_surface, (self.width - 300, 40))
        self.screen.blit(position_surface, (self.width - 300, 70))
        
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
            for i,body in enumerate(self.system.bodies):
                # Draw trails
                for trail_pos in body.trail:
                    screen_pos = self.world_to_screen(trail_pos)
                    pygame.draw.circle(self.screen, body.color, screen_pos, 1)
                
                # Draw bodies
                #scaled_position = body.position.copy()
                #if (i > 0):
                #    scaled_position/=i
                #pos = self.world_to_screen(scaled_position)
                pos = self.world_to_screen(body.position)
                # Size based on mass (logarithmic scale)
                radius = max(2, int(np.log10(body.mass) - 20))
                if body == self.system.bodies[0]:
                    radius = 1
                pygame.draw.circle(self.screen, body.color, pos, radius)
            self.render_text() 
            pygame.display.flip()
            self.clock.tick(600000)

if __name__ == "__main__":
    sim = Simulator()
    sim.run()
