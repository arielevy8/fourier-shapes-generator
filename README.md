# FourierShapesGenerator
Generation of novel shapes based on fourier descriptors for category learning experiments. 

This repository allows users to create novel shapes using Fourier descriptors, following the method presented by _____. This method is quite helpful for category-learning tasks in cognitive experiments, as it enables presenting participants with completely novel shapes, yet manipulating these shapes according to a strict rule set. Examples of using this method in category learning tasks can be found here, here, and here. This repo also allows users to create 2-dimensional subspaces of shapes, and to sample shapes from this subspace.

## Here are some examples of how to use the code:
### FourierShape class
The methods in this class allows for the generation of a shpae based on fourier descriptors.

    check = FourierShape(num_descriptors = 5)

    check.cumbend_to_points()

    check.plot_shape()

    plt.show()

 <img src="images/random%20shape%205.png" width="250" height="250">

This code section plots a random shape that is based on 5 Fourier descriptors. Increasing the number of Fourier descriptors adds complexity to the shape.

You can also specify the value of each Fourier descriptor, as follows.

    check = FourierShape(num_descriptors = 5, descriptor_amp=[0,0,0,0,0])

    check.cumbend_to_points()

    check.plot_shape()

    plt.show()

 <img src="images/descriptor_amp00000.png" width="250" height="250">
or 

    check = FourierShape(num_descriptors = 5, descriptor_amp=[0,0,0,1,0])

    check.cumbend_to_points()

    check.plot_shape()

    plt.show()

<img src="images/descriptor_amp00010.png" width="250" height="250">

### ShpaeSubspace class

This class use 3 points on an N-dimentional shapes space in order to
define a 2-dimentional shape subspace, which useful for category learning tasks.
This is done via the gram-schmidt process.
    
