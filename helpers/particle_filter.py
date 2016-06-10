import numpy.random


def generate_gaussian_noise(mu=0, sigma=1):
    return numpy.random.normal(mu, sigma)


class Particle:

    def __init__(self, x, y, w, h, sigma):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.sigma = sigma

    def propagate(self):
        self.x += generate_gaussian_noise(0, self.sigma)
        self.y += generate_gaussian_noise(0, self.sigma)
        self.w += generate_gaussian_noise(0, sqrt(2)*self.sigma)
        self.h += generate_gaussian_noise(0, sqrt(2)*self.sigma)


class ParticleFilter:

    def __init__(self, num_particles, x, y, w, h, sigma):
        self.num_particles = num_particles
        self.particles = list()
        for i in xrange(self.num_particles):
            self.particles.append(Particle(x, y, w, h, sigma))

    def propagate(self):
        [particle.propagate() for particle in self.particles]
