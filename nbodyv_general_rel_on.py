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
        self.vels = []
        self.pericount = 0
        self.peri_angle = 0
        self.steps_since_peri = 0

class GravitationalSystem:
    #units: m, kg, s
    G = 6.67430e-11  # gravitational constant
    earth_sun_max_distance = 0
    RELATIVITY_ON = True 
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
                #Body(mass_earth, [pdist_earth, 0 , zmax_earth], [0, pvel_earth, 0], (0, 150, 245)), 
                #Body(mass_moon, [pdist_moon, 0, zmax_earth], [0, pvel_moon, 0], (255, 255, 255)), 
                Body(mass_mercury, [pdist_mercury, 0, zmax_mercury], [0, pvel_mercury, 0], (255, 100, 100)),  # 1 - RED
                Body(mass_mercury, [pdist_mercury, 0, zmax_mercury], [0, pvel_mercury, 0], (100, 100, 255)), # 2 - BLUE
                #Body(mass_venus, [pdist_venus, 0, zmax_venus], [0, pvel_venus, 0], (255, 128, 129)), 
                #Body(mass_mars, [pdist_mars, 0, zmax_mars], [0, pvel_mars, 0], (255, 0, 0)), 
                #Body(mass_phobos, [pdist_phobos, 0, zmax_mars], [0, pvel_phobos, 0], (255, 255, 255)), 
                #Body(mass_deimos, [pdist_deimos, 0, zmax_mars], [0, pvel_deimos, 0], (255, 255, 255)), 
                #Body(mass_jupiter, [pdist_jupiter, 0, zmax_jupiter], [0, pvel_jupiter, 0], (255, 128, 0)), 
                #Body(mass_europa, [pdist_europa, 0, zmax_jupiter], [0, pvel_europa, 0], (255, 255, 255)), 
                #Body(mass_ganymede, [pdist_ganymede, 0, zmax_jupiter], [0, pvel_ganymede, 0], (128, 128, 128)), 
                #Body(mass_io, [pdist_io, 0,0], [0, pvel_io, zmax_jupiter], (255, 128, 244)), 
                #Body(mass_saturn, [pdist_saturn, 0, zmax_saturn], [0, pvel_saturn, 0], (100,100, 0)), 
                #Body(mass_uranus, [pdist_uranus, 0, zmax_uranus], [0, pvel_uranus, 0], (100, 100, 100)), 
                #Body(mass_neptune, [pdist_neptune, 0, zmax_neptune], [0, pvel_neptune, 0], (0, 10, 255)), 
        ]
        # SIM PARAMS 
        # time step in seconds
        self.dt = 1000
        # pixels per million km
        self.scale = 2e8

    def calculate_acceleration(self, bodies):
        c = 299792458.0
        accelerations = [np.array([0.0, 0.0, 0.0]) for _ in bodies]
        i = 0
        for j in range(i+1, len(bodies)):
            r = bodies[j].position - bodies[i].position
            r_mag = np.linalg.norm(r)
            force_mag = self.G * bodies[i].mass * bodies[j].mass / (r_mag ** 2)
            if(j == 2): #only applying correction to planet 2 BLUE
                v_i = np.linalg.norm(bodies[i].velocity)
                v_j = np.linalg.norm(bodies[j].velocity) #exaggerated for demo purposes
                pn2_correction = 1.0 + (
                    1e4*((4 * self.G * (bodies[i].mass + bodies[j].mass)) / (r_mag * c**2) 
                     - (4 * self.G**2 * (bodies[i].mass * bodies[j].mass) / (r_mag**3 * c**2)) 
                     - ((v_i**2 + v_j**2) / (2 * c**2))))
                force_mag *= pn2_correction
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
            #body.vels.append(body.velocity)
            #if i != 0 and len(body.vels) > 3:
            #   body.vels.pop(0)
            #   #print(np.linalg.norm(body.vels))
            #   if np.linalg.norm(body.vels[0]) < np.linalg.norm(body.vels[1]) and (np.linalg.norm(body.vels[2]) < np.linalg.norm(body.vels[1])):
            #           print ("fine perihelion")
            #mercuries perihelion detect and track
            if i == 1 or i == 2:
                body.steps_since_peri += 1
                error_margin = 1.0
                if ((5.897e4 - np.linalg.norm(body.velocity) < error_margin) and (body.steps_since_peri > 1e3)):
                    body.pericount+= 1
                    body.steps_since_peri = 0
                    body.peri_angle = math.atan2(body.position[1] ,body.position[0])
                    print("BODY ", i, " PERI", body.pericount, " ", body.peri_angle, " POS: ", body.position)
                    trailpos = body.position.copy()
                    trailpos*=[(1+i/10),(1+i/10),1]
                    body.trail.append(trailpos)
            if len(body.trail) > 1:
                body.trail.pop(0)
        

class Simulator:
    def __init__(self, width=900, height=800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("mercury precession gr demo")
        self.system = GravitationalSystem(3)
        self.clock = pygame.time.Clock()
        
    def world_to_screen(self, position):
        # Convert world coordinates to screen coordinates
        screen_x = position[0] / self.system.scale + self.width // 2
        screen_y = position[1]/ self.system.scale + self.height // 2
        return int(screen_x), int(screen_y)

    def render_text(self):
        font = pygame.font.Font(None, 24)
        
        precession_text = ("RED: mercury newtonian")
        precession_surface = font.render(precession_text, True, (255, 200, 200))
        self.screen.blit(precession_surface, (self.width - 230, 20)) 
        precession_text = ("BLUE: mercury relativistic")
        precession_surface = font.render(precession_text, True, (200, 200, 255))
        self.screen.blit(precession_surface, (self.width - 230, 50)) 
        
        angle1_text = f"Last Perihelion Angle (Red - nonrel): {math.degrees(self.system.bodies[1].peri_angle):.2f}°"
        angle2_text = f"Last Perihelion Angle (Blue - rel): {math.degrees(self.system.bodies[2].peri_angle):.2f}°"
        angle1_surface = font.render(angle1_text, True, (255, 255, 255))
        angle2_surface = font.render(angle2_text, True, (255, 255, 255))
        self.screen.blit(angle1_surface, (20, 20))  
        self.screen.blit(angle2_surface, (20, 50))  
        
    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            self.system.verlet_step()
            self.screen.fill((0, 0, 0))
            for i,body in enumerate(self.system.bodies):
                # Draw trails
                for t, trail_pos in enumerate(body.trail):
                    screen_pos = self.world_to_screen(trail_pos)
                    pygame.draw.circle(self.screen, body.color, screen_pos, 1)
                    #if t == 1:
                    #    pygame.draw.circle(self.screen, (255,0,0), screen_pos, 1)
                    #elif t == 2:
                    #    pygame.draw.circle(self.screen, (0,255,0), screen_pos, 1)
                    #else:
                    #    pygame.draw.circle(self.screen, (100,100,255), screen_pos, 1)
                
                # Draw bodies
                #scaled_position = body.position.copy()
                #if (i > 0):
                #    scaled_position/=i
                #pos = self.world_to_screen(scaled_position)
                bodypos = body.position.copy()
                #if (i == 1):
                #    bodypos*=[-1,1,1]
                pos = self.world_to_screen(bodypos)   
                # Size based on mass (logarithmic scale)
                radius = max(2, int(np.log10(body.mass) - 20))
                if i != 0:
                    radius = 3
                pygame.draw.circle(self.screen, body.color, pos, radius)
            self.render_text() 
            pygame.display.flip()
            self.clock.tick(500000)

if __name__ == "__main__":
    sim = Simulator()
    sim.run()
