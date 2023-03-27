typedef struct {
    uchar red;
    uchar green;
    uchar blue;
} Pixel;

__kernel void gaussianFilter(__global Pixel* image, __global Pixel* bluredData, __global double* kernelGauss, __global int* args)
{
    int globalID = get_global_id(0);

    int kernelGaussSize = args[0];
    int kernelRadius = args[1];
    int width = args[2];
    int height = args[3];

    int j = globalID % width;
    int i = globalID / width;

    double sumRed = 0;
    double sumGreen = 0;
    double sumBlue = 0;

    for (int k = -kernelRadius; k <= kernelRadius; k++) {
        for (int l = -kernelRadius; l <= kernelRadius; l++) {
            if (i + k < 0 || i + k >= height || j + l < 0 || j + l >= height) {
                sumRed += kernelGauss[(k + kernelRadius) * kernelGaussSize + l + kernelRadius] * image[i * width + j].red;
                sumGreen += kernelGauss[(k + kernelRadius) * kernelGaussSize + l + kernelRadius] * image[i * width + j].red;
                sumBlue += kernelGauss[(k + kernelRadius) * kernelGaussSize + l + kernelRadius] * image[i * width + j].red;
            } else {
                sumRed += kernelGauss[(k + kernelRadius) * kernelGaussSize + l + kernelRadius] * image[(i + l) * width + j + k].red;
                sumGreen += kernelGauss[(k + kernelRadius) * kernelGaussSize + l + kernelRadius] * image[(i + l) * width + j + k].green;
                sumBlue += kernelGauss[(k + kernelRadius) * kernelGaussSize + l + kernelRadius] * image[(i + l) * width + j + k].blue;
            }
        }
    }

    bluredData[globalID].red = convert_uchar_rte(sumRed);
    bluredData[globalID].green = convert_uchar_rte(sumGreen);
    bluredData[globalID].blue = convert_uchar_rte(sumBlue);
}