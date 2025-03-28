import numpy as np
import matplotlib.pyplot as plt
from .FourierShape import FourierShape
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class ShapeSubspace(FourierShape):
    """
    This class use 3 points on an N-dimentional shapes space in order to
    define a 2-dimentional shape subspace, which useful for category learning tasks.
    This is done via the gram-schmidt process.
    """

    def __init__(self, n_descriptors, phase_offset=0, amp_bound=(-0.8,0.8)):
        """
        :param num_descriptor: The number of fourier descriptors used to create
        the shape. Increasing this number adds complexity to the shpae.
        :param descriptor_phase: the phase of each fourier descriptor.
        :param phase_offset: a general phase offset for all fourier descriptors.
        :param amp_bound: the bounds of the amplitude of each fourier descriptor.
        """
        super().__init__(n_descriptors, phase_offset=phase_offset, amp_bound=amp_bound)
        # Store n_descriptors for use in other methods
        self.num_descriptors = n_descriptors

    def gs_cofficient(self, v1, v2):
        """
        Calculate the Gram-Schmidt coefficient for projection
        """
        return np.dot(v2, v1) / np.dot(v1, v1)

    def multiply(self, cofficient, v):
        return map((lambda x: x * cofficient), v)

    def proj(self, v1, v2):
        """
        Project v2 onto v1
        """
        return self.gs_cofficient(v1, v2) * v1

    def gs(self, old):
        """
        This method implements Gram-Schmidt orthonormalization to create
        subspace.
        :param old: Array of 2 direction vectors from the point 1 to points 2 and 3
        returns: new, an array of orthogonal vectors to span the subspace
        """
        new = []
        for i in range(len(old)):
            cur_vec = old[i].copy()
            for inY in new:
                proj_vec = self.proj(inY, cur_vec)
                cur_vec = cur_vec - proj_vec
            # Normalize the vector
            norm = np.linalg.norm(cur_vec)
            if norm > 0:
                cur_vec = cur_vec / norm
            new.append(cur_vec)
        return np.array(new)

    def generate_subspace(self, factor1=1, factor2=None, point1=None, point2=None, point3=None):
        """
        This function generates a subspace that contain 3 points
        using Gram-Schmidt orthonormalization. 
        :param factor: float representing the size of the eventual orthogonal vector.
        when increased, the edge shapes will be far from the base shape (point1)
        :param point1, point 2, point3: lists with the length of 'num_descriptors', representing the
        amplitude of each fourier descriptor for each of the 3 points. If not stated,
        it is randomally determined. 
        """
        if not point1:
            point1 = np.random.uniform(-1, 1, self.num_descriptors)
        if not point2:
            point2 = np.random.uniform(-1, 1, self.num_descriptors)
        if not point3:
            point3 = np.random.uniform(-1, 1, self.num_descriptors)
        if len(point1) != self.num_descriptors or len(point2) != self.num_descriptors or len(
                point3) != self.num_descriptors:
            raise ValueError('length of the points list should' +
                             'be equal to the number of the descriptors')
        
        # Convert to numpy arrays if they aren't already
        point1 = np.array(point1)
        point2 = np.array(point2)
        point3 = np.array(point3)
        
        self.point1 = point1
        self.vec12 = point2 - point1  # direction vector from point 1 to point 2
        self.vec13 = point3 - point1  # direction vector from point 3 to point 1
        
        # Apply Gram-Schmidt orthonormalization
        gs_coeffs = self.gs(np.array([self.vec12, self.vec13]))
        gs1, gs2 = gs_coeffs[0, :], gs_coeffs[1, :]
        
        # Scale the orthogonal vectors by the factors
        gs1 = factor1 * gs1
        if factor2 is not None:
            gs2 = factor2 * gs2
        else:
            gs2 = factor1 * gs2
        
        self.gs1, self.gs2 = gs1, gs2
        self.end1 = self.point1 + gs1
        self.end2 = self.point1 + gs2

    def plot_subspace(self):
        """
        This function demonstrates the location of thesubspace plane on the original
        3d space.
        Note that this function only work on 3d shapes space (i.e., when num_descriptors
        equal to 3)
        """
        if self.num_descriptors != 3:
            raise ValueError('Number of descriptors must be equal to 3 in order to plot subspace')
        else:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot([self.point1[0], self.end2[0]], [self.point1[1], self.end2[1]], [self.point1[2], self.end2[2]],
                    label='dim1',linewidth = 4)
            ax.plot([self.point1[0], self.end1[0]], [self.point1[1], self.end1[1]], [self.point1[2], self.end1[2]],
                    label='dim2',linewidth = 4)
            x = [self.point1[0], self.end1[0], self.end2[0]]
            y = [self.point1[1], self.end1[1], self.end2[1]]
            z = [self.point1[2], self.end1[2], self.end2[2]]
            verts = list(zip(x, y, z))
            coll = Poly3DCollection(verts)
            coll.set_color('grey')
            coll.set_alpha(0.5)
            ax.add_collection3d(coll)
            plt.legend()
            plt.show()

    def plot_shapes_grid(self, num_levels):
        """
        This function creates a grid of shapes in the the subscpace
        and plot it.
        :param num_levels: a natural number that states how many morph
        levels the grid will contain for each dimention.
        """
        first_dim = np.linspace(self.point1, self.end1,
                                num_levels)  # a shape dimention that vary from the first random shpae to the second
        second_dim = np.linspace(self.point1, self.end2,
                                 num_levels)  # a shape dimention that vary from the first random shpae to the third
        counter = 0
        for dim_1 in range(first_dim.shape[0]):
            for dim_2 in range(second_dim.shape[0]):
                coeffs = (first_dim[dim_1] + second_dim[dim_2]) / 2
                self.descriptor_amp = self.short_to_full(coeffs)
                self.descriptors_to_shape()
                xt, yt = self.points[:, 0], self.points[:, 1]
                ax = plt.subplot(first_dim.shape[0], second_dim.shape[0], counter + 1)
                ax.plot(xt, yt)
                ax.fill_between(xt, yt)
                ax.axis('off')
                counter += 1
        plt.show()

    def save_shapes_grid(self, num_levels, path):
        """
        This function creates a grid of shapes in the the subscpace
        and save it in a chosen directory.
        :param num_levels: a natural number that states how many morph
        levels the grid will contain for each dimention.
        :param_path: the path in which you wish to sace the files.
        """
        first_dim = np.linspace(self.point1, self.end1,
                                num_levels)  # a shape dimention that vary from the first random shpae to the second
        second_dim = np.linspace(self.point1, self.end2,
                                 num_levels)  # a shape dimention that vary from the first random shpae to the third
        counter = 0
        dim_1_counter = -(num_levels // 2)
        dim_2_counter = -(num_levels // 2)
        for dim_1 in range(first_dim.shape[0]):
            for dim_2 in range(second_dim.shape[0]):
                coeffs = (first_dim[dim_1] + second_dim[dim_2]) / 2
                self.descriptor_amp = self.short_to_full(coeffs)
                self.descriptors_to_shape()
                xt, yt = self.points[:, 0], self.points[:, 1]
                plt.figure()
                plt.plot(xt, yt)
                plt.fill_between(xt, yt)
                plt.savefig(path + '/' + str(dim_1_counter) + '_' + str(dim_2_counter) + '.jpg')
                plt.close()
                counter += 1
                dim_2_counter += 1
            dim_1_counter += 1
            dim_2_counter = -(num_levels // 2)

    def sample_from_subspace(self, num_shapes, path, dim1_mean=None, dim1_sd=None
                             , dim2_mean=None, dim2_sd=None, plot_hist=True):

        """
        This function allows sampling from the 2 dimentional shape subspace 
        defined by this class, using normal distribution with predefined
        mean and sd, or rather uniform distribution. You can also sample from 
        one dimention using normal dist and from the other dimention using uniform.
        :param num shapes: number of shpaes you wish to create
        :param path: directory to export shapes images
        :params dim1_mean, dim1_sd, dim2_mean, dim2_sd: parameters for the normal distribution
        from which shapes are sampled. if not stated for one or both dims, shapes
        are sampled from the uniform distribution in this dim.
        
        """
        dim1_list = []
        dim2_list = []
        for i in range(num_shapes):
            if dim1_sd:
                cur_loc_dim_1 = np.random.normal(dim1_mean, dim1_sd)
            else:
                cur_loc_dim_1 = np.random.uniform(0, 1)
            if dim2_sd:
                cur_loc_dim_2 = np.random.normal(dim2_mean, dim2_sd)
            else:
                cur_loc_dim_2 = np.random.uniform(0, 1)
            dim1_list.append(cur_loc_dim_1)
            dim2_list.append(cur_loc_dim_2)
            descriptors_dim_2 = self.point1 + cur_loc_dim_2 * self.gs2
            descriptors_dim_1 = self.point1 + cur_loc_dim_1 * self.gs1
            descriptors = (descriptors_dim_1 + descriptors_dim_2) / 2
            self.descriptor_amp = self.short_to_full(descriptors)
            self.descriptors_to_shape()
            xt, yt = self.points[:, 0], self.points[:, 1]
            plt.figure()
            plt.plot(xt, yt)
            plt.fill_between(xt, yt)
            plt.savefig(path + '/shape_' + str(round(cur_loc_dim_1, 3)) + '_' + str(round(cur_loc_dim_2, 3)) + '.jpg')
            plt.close()
        if plot_hist:
            plt.hist2d(dim2_list, dim1_list, bins=[np.linspace(min(dim2_list), max(dim2_list), num_shapes // 10),
                                                   np.linspace(min(dim1_list), max(dim1_list), num_shapes // 10)])
            plt.show()



if __name__ == "__main__":
    sub = ShapeSubspace(4,phase_offset=65,amp_bound=(-0.9,0.9))
    sub.generate_subspace(2.5)
    sub.plot_shapes_grid(8)
    #sub.plot_subspace()
    # Check orthogonality
    dot_product = np.dot(sub.gs1, sub.gs2)
    print(f"Dot product between orthogonal vectors: {dot_product}")  # Should be very close to 0
    # Check dot product between subspace vectors
    v1 = sub.end1 - sub.point1  # Vector from point1 to end1 
    v2 = sub.end2 - sub.point1  # Vector from point1 to end2
    subspace_dot = np.dot(v1, v2)
    print(f"\nDot product between subspace vectors: {subspace_dot}")
    
    # Also check angles
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    cos_angle = subspace_dot / (v1_norm * v2_norm)
    angle = np.arccos(cos_angle) * 180 / np.pi
    print(f"Angle between subspace vectors: {angle:.2f} degrees")