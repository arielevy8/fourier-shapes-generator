import numpy as np
import matplotlib.pyplot as plt


class FourierShape(object):
    """
    This class contain methods to generate a shpae based on a set
     of N descriptors.
    """

    def __init__(self, num_descriptors, descriptor_amp=None, descriptor_phase=None):
        """
        :param num_descriptor: The number of fourier descriptors used to create
        the shape. Increasing this number ads complexity to the shpae.
        :param descriptor_amp: a list with the length of 'num_descriptors', representing the
        amplitude of each fourier descriptor. If not stated,
        it is randomally determined.
        :param descriptor_phase: Phase of each fourier descriptor. If not stated, 
        all phases are equal to 0
        """
        self.num_descriptors = num_descriptors

        if not descriptor_phase:
            descriptor_phase = np.zeros(num_descriptors)
        elif len(descriptor_phase) != num_descriptors:
            raise ValueError('length of the descriptor_phase list should be equal to the number of the descriptors')
        descriptor_phase = self.short_to_full(descriptor_phase)
        self.descriptor_phase = descriptor_phase

        if not descriptor_amp:
            descriptor_amp = np.random.uniform(-1, 1, num_descriptors)
        elif len(descriptor_amp) != num_descriptors:
            raise ValueError('length of the descriptor_amp list should be equal to the number of the descriptors')
        descriptor_amp = self.short_to_full(descriptor_amp)
        self.descriptor_amp = descriptor_amp

    def full_to_short(self, full):
        return full[0, 2, 4, 6]

    def short_to_full(self, short):
        return np.ravel(list(zip(short, [0] * len(short))))

    def cumbend(self, t):
        """
        This method calculates the next theta angle for 
        the timepoint t.
        """
        theta = -t
        for freq in range(len(self.descriptor_amp)):
            amp = self.descriptor_amp[freq]
            phase = (self.descriptor_phase[freq] / 360) * 2 * np.pi
            theta += amp * np.cos(freq * t - phase)
        return theta

    def cumbend_to_points(self, steps=1000):
        """
        This method uses the descriptors to calculate the point set, based
        on the process described by __
        :param steps: Number of time-points in the 
        """
        cur_point = complex(real=0, imag=0)
        points = []
        for t in np.linspace(0, 2 * np.pi, steps):
            bend_angle = self.cumbend(t)
            following_point = cur_point + complex(real=np.cos(bend_angle), imag=np.sin(bend_angle))
            mini_list = [np.real(cur_point), np.imag(cur_point)]
            points.append(mini_list)
            cur_point = following_point
        self.points = np.array(points)

    def plot_shape(self, color=None):
        plt.fill_between(self.points[:, 0], self.points[:, 1], color=color, edgecolor="none")
        plt.axis('off')
