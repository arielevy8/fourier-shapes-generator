import numpy as np
import matplotlib.pyplot as plt
from FourierShape import FourierShape

class ShapeSubspace (FourierShape):
    """
    This class use 3 points on an N-dimentional shapes space in order to
    define a 2-dimentional shape subspace, which useful for category learning tasks.
    This is done via the gram-schmidt process.
    """
    def __init__(self,num_descriptors):
        """
        :param num_descriptor: The number of fourier descriptors used to create
        the shape. Increasing this number adds complexity to the shpae.
        """
        super().__init__(num_descriptors)
        self.num_descriptors = num_descriptors
        self.descriptor_phase = np.zeros (num_descriptors)
        self.descriptor_phase = self.short_to_full(self.descriptor_phase) 
    
    def gs_cofficient(self,v1, v2):
        return np.dot(v2, v1) / np.dot(v1, v1)

    def multiply(self,cofficient, v):
        return map((lambda x : x * cofficient), v)

    def proj(self,v1, v2):
        return self.multiply(self.gs_cofficient(v1, v2) , v1)

    def gs(self,old):
        """
        This method implements gram-schmidt orthonormalizatin to create
        subspace.
        :param old: Array of 2 direction vectors from the point 1 to points 2 and 3
        returns: new, an array of orthogonal vectors to span the subspace
        """
        new = []
        for i in range(len(old)):
            cur_vec = old[i]
            for inY in new :
                proj_vec = self.proj(inY, old[i])
                cur_vec = list(map(lambda x, y : x - y, cur_vec, proj_vec))
            new.append(cur_vec)
        return new

    def generate_subspace(self,factor,point1=None,point2=None,point3=None):
        """
        This function generates a subspace that contain 3 points
        using gram-schmidt orthomormalization. 
        :param factor: float representing the size of the eventual orthogonal vector.
        when increased, the edge shapes will be far from the base shape (point1)
        :param point1, point 2, point3: lists with the length of 'num_descriptors', representing the
        amplitude of each fourier descriptor for each of the 3 points. If not stated,
        it is randomally determined. 
        """
        if not point1:
            point1 = np.random.uniform (-1,1,self.num_descriptors)
        if not point2:
            point2 = np.random.uniform (-1,1,self.num_descriptors)
        if not point3:
            point3 = np.random.uniform (-1,1,self.num_descriptors)
        if len(point1) != self.num_descriptors or len(point2) != self.num_descriptors or len(point3) != self.num_descriptors  :
            raise ValueError ('length of the points list should'+
            'be equal to the number of the descriptors')
        self.point1 = point1
        self.vec12 = point2-point1 #direction vector from point 1 to point 2
        self.vec13 = point3-point1 # direction vector from point 3 to poin 1
        gs_coeffs = np.array(self.gs(np.array([self.vec12,self.vec13])))
        gs1,gs2 = gs_coeffs[0,:],gs_coeffs[1,:]
        gs1 = factor*gs1/np.linalg.norm(gs1)
        gs2 = factor*gs2/np.linalg.norm(gs2)
        self.end1 = self.point1+gs1
        self.end2 = self.point1+gs2
    
    def plot_subspace (self):
        """
        """
        pass
    
    def plot_shapes_grid (self,num_levels):
        """
        This function creates a grid of shapes in the the subscpace
        and plot it.
        :param num_levels: a natural number that states how many morph
        levels the grid will contain for each dimention.
        """
        first_dim = np.linspace(self.point1,self.end1,num_levels) #a shape dimention that vary from the first random shpae to the second 
        second_dim = np.linspace(self.point1,self.end2,num_levels) #a shape dimention that vary from the first random shpae to the third 
        counter = 0
        for dim_1 in range(first_dim.shape[0]):
            for dim_2 in range(second_dim.shape[0]):
                coeffs = (first_dim[dim_1]+second_dim[dim_2])/2
                self.descriptor_amp = self.short_to_full(coeffs)
                self.cumbend_to_points()
                xt,yt = self.points[:,0],self.points[:,1]
                ax = plt.subplot(first_dim.shape[0],second_dim.shape[0],counter+1)
                ax.plot (xt,yt)
                ax.fill_between(xt,yt)
                ax.axis('off')
                counter+=1
        plt.show()

    def save_shapes_grid (self,num_levels,path):
        """
        This function creates a grid of shapes in the the subscpace
        and save it in a chosen directory.
        :param num_levels: a natural number that states how many morph
        levels the grid will contain for each dimention.
        :param_path: the path in which you wish to sace the files.
        """
        first_dim = np.linspace(self.point1,self.end1,num_levels) #a shape dimention that vary from the first random shpae to the second 
        second_dim = np.linspace(self.point1,self.end2,num_levels) #a shape dimention that vary from the first random shpae to the third 
        counter = 0
        dim_1_counter = -(num_levels//2)
        dim_2_counter = -(num_levels//2)
        for dim_1 in range(first_dim.shape[0]):
            for dim_2 in range(second_dim.shape[0]):
                coeffs = (first_dim[dim_1]+second_dim[dim_2])/2
                self.descriptor_amp = self.short_to_full(coeffs)
                self.cumbend_to_points()
                xt,yt = self.points[:,0],self.points[:,1]
                plt.figure()
                plt.plot (xt,yt)
                plt.fill_between(xt,yt)
                plt.savefig(path+'/'+str(dim_1_counter)+'_'+str(dim_2_counter)+'.jpg')
                plt.close()
                counter+=1
                dim_2_counter+=1
            dim_1_counter+=1
            dim_2_counter = -(num_levels//2)

    def sample_from_subspace (self, num_shapes,path, dim1_mean = None,dim1_sd = None
                                ,dim2_mean = None, dim2_sd = None, plot_hist = True):
        pass


sub = ShapeSubspace(4)
sub.generate_subspace(1.5)
sub.plot_shapes_grid(9)

