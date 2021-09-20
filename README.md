# TextureSegmentationOfImages
Texture segmentation of images based on a statistical approach.
Segmentation is performed based on the calculation of two statistics: R and the third moment.

R = 1 - (1/ (1 + sigma^2(z))
sigma^2(z) = M2(z)

Mn(z) = sum[(zi - m)^2 * p(zi)], i = 0 .. L-1

z is a random variable corresponding to the brightness of the image elements.
p (z), i = 0,1,2..L-1 - its histogram, where L denotes the number of different brightness levels.
m - mean value of z

M3(z) = sum[(zi - m)^3 * p(zi)], i = 0 .. L-1
