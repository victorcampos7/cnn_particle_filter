import numpy.random
import random
import numpy as np
from math import sqrt
import cv2


def generate_gaussian_noise(mu=0, sigma=1):
    return numpy.random.normal(mu, sigma)


def compute_features(current_features, x, y, w, h, combination_method='avg'):
    cropped_feature_map = current_features[:, y:y+h, x:x+w]
    if combination_method == 'avg':
        return np.mean(np.mean(cropped_feature_map, axis=1), axis=1)
    elif combination_method == 'max':
        return np.amax(np.amax(cropped_feature_map, axis=1), axis=1)
    else:
        raise ValueError('Invalid combination method')


def resize_feature_map(feature_map, size_x, size_y):
    c = np.shape(feature_map)[0]
    new_size = (c, size_y, size_x)
    return cv2.resize(feature_map, new_size, interpolation=cv2.INTER_NEAREST)


class Particle:

    def __init__(self, x, y, w, h, sigma, img_dims, noise_distr='gaussian', combination_method='avg'):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.sigma = sigma
        # img_dims = (h,w) like in numpy
        self.max_x = img_dims[1] - 1
        self.max_y = img_dims[0] - 1
        self.noise_distr = noise_distr
        self.combination_method = combination_method

    def generate_noise(self, mu=0, sigma=1):
        if self.noise_distr == 'gaussian':
            return generate_gaussian_noise(mu, sigma)
        elif self.noise_distr == 'uniform':
            return random.uniform(-sigma, sigma)
        else:
            raise ValueError('Invalid noise distribution')

    def propagate(self):
        self.x += int(max(min(self.generate_noise(sigma=self.sigma), self.max_x), 0))
        self.y += int(max(min(self.generate_noise(sigma=self.sigma), self.max_y), 0))
        self.w += int(max(min(self.generate_noise(sigma=sqrt(2)*self.sigma), self.max_x-self.x), 0))
        self.h += int(max(min(self.generate_noise(sigma=sqrt(2)*self.sigma), self.max_y-self.y), 0))

    def compute_features(self, current_features):
        return compute_features(current_features, self.x, self.y, self.w, self.h, self.combination_method)

    def compute_weight(self, current_features, model):
        particle_features = self.compute_features(current_features)
        dist = np.linalg.norm(model - particle_features)
        # Weight ~ inverse euclidean distance
        return 1./(1. + dist)

    def get_values(self):
        return self.x, self.y, self.w, self.h


class ParticleFilter:

    def __init__(self, num_particles, x, y, w, h, sigma, img_dims, noise_distr='gaussian', combination_method='avg'):
        self.num_particles = num_particles
        self.particles = list()
        for i in xrange(self.num_particles):
            self.particles.append(Particle(x, y, w, h, sigma, img_dims, noise_distr, combination_method))
        self.current_features = None
        self.model = None
        self.weights = (1./self.num_particles)*np.ones(self.num_particles, dtype=np.float32)
        self.img_dims = img_dims

    def set_model(self, model):
        self.model = resize_feature_map(model, self.img_dims[1], self.img_dims[0])

    def set_current_features(self, current_features):
        self.current_features = resize_feature_map(current_features, self.img_dims[1], self.img_dims[0])

    def propagate(self):
        [particle.propagate() for particle in self.particles]

    def compute_weights(self):
        # Compute weights
        for i, particle in enumerate(self.particles):
            self.weights[i] = particle.compute_weight(self.current_features, self.model)
        # Normalize weights
        self.weights = self.weights/np.sum(self.weights)

    def resample_particles(self):
        new_indices = list()
        # Compute CDF
        cdf = [0.] + [sum(self.weights[:i+1]) for i in range(self.num_particles)]
        # Sample using random values and finding where they belong in the CDF
        u0, j = numpy.random.random(), 0
        for u in [(u0 + i) / n for i in range(n)]:
            while u > cdf[j]:
                j += 1
            new_indices.append(j - 1)
        # Re-sample particles
        self.particles = self.particles[new_indices]
        # Re-sample weights (optional, since they will be recomputed again in the next iteration)
        self.weights = (1. / self.num_particles) * np.ones(self.num_particles, dtype=np.float32)

    def _get_all_particles(self):
        x_values = np.zeros(self.num_particles)
        y_values = np.zeros(self.num_particles)
        w_values = np.zeros(self.num_particles)
        h_values = np.zeros(self.num_particles)
        for i, particle in enumerate(self.particles):
            x_values[i], y_values[i], w_values[i], h_values[i] = particle.get_values()
        return x_values, y_values, w_values, h_values

    def compute_bounding_box(self):
        # Get all the values
        x_values, y_values, w_values, h_values = self._get_all_particles()
        # Compute the expectation as the weighted sum
        x = int(np.sum(self.weights.T * x_values, axis=0))
        y = int(np.sum(self.weights.T * y_values, axis=0))
        w = int(np.sum(self.weights.T * w_values, axis=0))
        h = int(np.sum(self.weights.T * h_values, axis=0))
        return x, y, w, h

    def track(self, current_features):
        # Set features for this frame
        self.set_current_features(current_features)
        # Propagate particles
        self.propagate()
        # Compute the weights for the new bounding boxes
        self.compute_weights()
        # Compute estimated bounding box for this frame
        x, y, w, h = self.compute_bounding_box()
        # Re-sample particles
        self.resample_particles()
        # Return estimated bounding box
        return x, y, w, h
