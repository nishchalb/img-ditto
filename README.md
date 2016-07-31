# img-ditto
img-ditto is a small python script with a command line interface that 
allows you to transfer the color style from one image to another. It works by 
using the [histogram matching](https://en.wikipedia.org/wiki/Histogram_matching)
technique on the channels of an image. Example outputs are shown below.

## Example Results
Transferring style from:

![ from image ](examples/ny_from.jpg)

To this image:

![to image](examples/ny_to.jpg)

Results in :
![New York result](examples/ny_lab.jpg)

Transferring style from:

![ from image ](examples/forest_from.jpg)

To this image:

![to image](examples/forest_to.jpg)

Results in :
![Forest result](examples/forest_lab.jpg)

Transferring style from:

![ from image ](examples/portrait_from.jpg)

To this image:

![to image](examples/portrait_to.jpg)

Results in :
![portrait result](examples/portrait_lab.jpg)

## Dependencies
img-ditto has a few dependencies. Before using img-ditto, the following python packages should also be installed:

1. [Numpy 1.8.2+] (http://www.numpy.org/)

2. [OpenCV 2.4.8+ with Python support] (http://opencv.org/)

3. [SciPy 0.13.3+](https://www.scipy.org/)

## Installation
Once the dependencies are installed, clone the repository. At this point, you can choose to either use img-ditto as a normal script, or you can make it work globally by adding the cloned repository to your PATH, or by moving img-ditto to somewhere in your path.

## Usage
```
usage: img-ditto [-h] [-o OUTPUT] [-s STRENGTH] [-c COLORSPACE]
                 from_im to [to ...]

Transfer the style from one image to another using histogram distributions

positional arguments:
  from_im               The image who's style you want to use
  to                    The image(s) you want to transfer the style to

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        The suffix to append to the output files
  -s STRENGTH, --strength STRENGTH
                        The strength of the transfer, should be between 0.0
                        and 1.0. Defaults to 0.8
  -c COLORSPACE, --colorspace COLORSPACE
                        The colorspace to use for the transformation.
                        Currently supported are rgb, hsv, and lab. Defaults to
                        lab.
```

