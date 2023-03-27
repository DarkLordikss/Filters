//#pragma GCC optimize("O3")
//#pragma GCC target("avx2")
#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define __CL_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define M_PI 3.14159265358979323846

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include "stb_image.h"
#include "stb_image_write.h"
#include "omp.h"
//#include <x86intrin.h>
#include <CL/cl.h>

const int MAX_SOURCE_SIZE = 10000;
const int kernelSize = 15;
double kernel[kernelSize][kernelSize];

struct Pixel {
    unsigned char red = 255;
    unsigned char green = 255;
    unsigned char blue = 255;
};

Pixel** parseData(unsigned char* data, int width, int height) {
	Pixel** parsedData = new Pixel*[height];
	int index = 0;

	for (int i = 0; i < height; i++) {
		parsedData[i] = new Pixel[width];
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			parsedData[i][j].red = data[index];
			parsedData[i][j].green = data[index + 1];
			parsedData[i][j].blue = data[index + 2];

			index = index + 3;
		}
	}

	return parsedData;
}

unsigned char* unparseData(Pixel** data, int width, int height) {
	unsigned char* unparsedData = new unsigned char[width * height * 3];
	int index = 0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			unparsedData[index++] = data[i][j].red;
			unparsedData[index++] = data[i][j].green;
			unparsedData[index++] = data[i][j].blue;
		}
	}

	return unparsedData;
}

Pixel** negativeFilter(Pixel** data, int width, int height) {
	Pixel** negativeData = new Pixel * [height];

    //#pragma omp parallel for num_threads(4)
	for (int i = 0; i < height; i++) {
		negativeData[i] = new Pixel[width];
	}

    //#pragma omp parallel for num_threads(4)
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			negativeData[i][j].red = 255 - data[i][j].red;
			negativeData[i][j].green = 255 - data[i][j].green;
			negativeData[i][j].blue = 255 - data[i][j].blue;
		}
	}

	return negativeData;
}

Pixel** negativeFilterVec(Pixel** data, int width, int height) {
    Pixel** negativeData = new Pixel * [height];
    for (int i = 0; i < height; i++) {
        negativeData[i] = new Pixel[width];
    }

    __m128i mask = _mm_set1_epi8(255);
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j += 4) {
            __m128i pixel = _mm_loadu_si128((__m128i*)(&data[i][j]));
            __m128i negPixel = _mm_sub_epi8(mask, pixel);
            _mm_storeu_si128((__m128i*)(&negativeData[i][j]), negPixel);
        }
    }

    return negativeData;
}

Pixel** negativeFilterCl(Pixel** data, int width, int height) {
    char* kernelSource;
    int size = width * height * 3;
    unsigned char* rawData = unparseData(data, width, height);
    unsigned char* rawNegativeData = new unsigned char[size];
    size_t kernelSize;
    FILE* fp;
    const char fileName[] = "Kernel.cl";

    cl_int err;
    cl_uint errNumPlatforms, errNumDevices;
    cl_platform_id platformID;
    cl_device_id deviceID;
    cl_context context;
    cl_command_queue commandQueue;
    cl_mem rawDataBuffer, rawDataNegativeBuffer;
    cl_program program;
    cl_kernel kernel;
    
    err = clGetPlatformIDs(1, &platformID, &errNumPlatforms);
    err = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, &errNumDevices);
    context = clCreateContext(NULL, errNumDevices, &deviceID, NULL, NULL, &err);
    commandQueue = clCreateCommandQueueWithProperties(context, deviceID, NULL, &err);
    rawDataBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * size, NULL, &err);
    rawDataNegativeBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * size, NULL, &err);

    fp = fopen(fileName, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    kernelSource = (char*)malloc(MAX_SOURCE_SIZE);
    kernelSize = fread(kernelSource, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, (const size_t*)&kernelSize, &err);
    err = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "negativeFilter", &err);

    err = clEnqueueWriteBuffer(commandQueue, rawDataBuffer, CL_TRUE, 0, sizeof(unsigned char) * size, rawData, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(commandQueue, rawDataNegativeBuffer, CL_TRUE, 0, sizeof(unsigned char) * size, rawNegativeData, 0, NULL, NULL);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &rawDataBuffer);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &rawDataNegativeBuffer);

    size_t globalWorkSize[1] = { size };

    auto start = std::chrono::steady_clock::now();
    
    err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    err = clEnqueueReadBuffer(commandQueue, rawDataNegativeBuffer, CL_TRUE, 0, sizeof(unsigned char) * size, rawNegativeData, 0, NULL, NULL);
    
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << static_cast<double>(elapsed) / 1000000 << " s" << std::endl;

    return parseData(rawNegativeData, width, height);
}

void GenerateGaussianKernel() {
    double sigma = 7.2;
    double r, s = 2.0 * sigma * sigma;

    int center = kernelSize / 2;

    double sum = 0.0;
    for (int x = 0; x < kernelSize; x++) {
        for (int y = 0; y < kernelSize; y++) {
            r = std::sqrt((x - center) * (x - center) + (y - center) * (y - center));
            kernel[x][y] = (std::exp(-(r * r) / s)) / (M_PI * s);
            sum += kernel[x][y];
        }
    }

    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            kernel[i][j] /= sum;
        }
    }
}

Pixel borderControl(int width, int height, Pixel** image, int i, int j, int k, int l) {
    if (i + k < 0 || i + k >= height || j + l < 0 || j + l >= height) {
        return image[i][j];
    }

    return image[i + k][j + l];
}

Pixel** gaussianFilter(Pixel** image, int width, int height) {
    const int kernelRadius = kernelSize / 2;
    Pixel** bluredData = new Pixel * [height];

    //#pragma omp parallel for num_threads(4)
    for (int i = 0; i < height; i++) {
        bluredData[i] = new Pixel[width];
    }

    //#pragma omp parallel for num_threads(4)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            double sumRed = 0;
            double sumGreen = 0;
            double sumBlue = 0;
            for (int k = -kernelRadius; k <= kernelRadius; k++) {
                for (int l = -kernelRadius; l <= kernelRadius; l++) {
                    sumRed += kernel[kernelRadius + k][kernelRadius + l] * 
                        borderControl(width, height, image, i, j, k, l).red;
                    sumGreen += kernel[kernelRadius + k][kernelRadius + l] * 
                        borderControl(width, height, image, i, j, k, l).green;
                    sumBlue += kernel[kernelRadius + k][kernelRadius + l] * 
                        borderControl(width, height, image, i, j, k, l).blue;
                }
            }
            bluredData[i][j].red = (unsigned char)sumRed;
            bluredData[i][j].green = (unsigned char)sumGreen;
            bluredData[i][j].blue = (unsigned char)sumBlue;
        }
    }

    return bluredData;
}

Pixel** gaussianFilterVec(Pixel** image, int width, int height) {
    const int kernelRadius = kernelSize / 2;
    Pixel** bluredData = new Pixel * [height];
    for (int i = 0; i < height; i++) {
        bluredData[i] = new Pixel[width];
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j += 4) { 
            __m256d sumRed = _mm256_setzero_pd();
            __m256d sumGreen = _mm256_setzero_pd();
            __m256d sumBlue = _mm256_setzero_pd();
            for (int k = -kernelRadius; k <= kernelRadius; k++) {
                for (int l = -kernelRadius; l <= kernelRadius; l++) {
                    __m256d kernelValue = _mm256_set1_pd(kernel[kernelRadius + k][kernelRadius + l]);
                    Pixel pixel0 = borderControl(width, height, image, i, j, k, l);
                    Pixel pixel1 = borderControl(width, height, image, i, j + 1, k, l);
                    Pixel pixel2 = borderControl(width, height, image, i, j + 2, k, l);
                    Pixel pixel3 = borderControl(width, height, image, i, j + 3, k, l);
                    __m256d pixelRed = _mm256_set_pd(pixel3.red, pixel2.red, pixel1.red, pixel0.red);
                    __m256d pixelGreen = _mm256_set_pd(pixel3.green, pixel2.green, pixel1.green, pixel0.green);
                    __m256d pixelBlue = _mm256_set_pd(pixel3.blue, pixel2.blue, pixel1.blue, pixel0.blue);
                    sumRed = _mm256_add_pd(sumRed, _mm256_mul_pd(kernelValue, pixelRed));
                    sumGreen = _mm256_add_pd(sumGreen, _mm256_mul_pd(kernelValue, pixelGreen));
                    sumBlue = _mm256_add_pd(sumBlue, _mm256_mul_pd(kernelValue, pixelBlue));
                }
            }

            double* reds = (double*)&sumRed;
            double* greens = (double*)&sumGreen;
            double* blues = (double*)&sumBlue;
            bluredData[i][j].red = (unsigned char)reds[0];
            bluredData[i][j + 1].red = (unsigned char)reds[1];
            bluredData[i][j + 2].red = (unsigned char)reds[2];
            bluredData[i][j + 3].red = (unsigned char)reds[3];
            bluredData[i][j].green = (unsigned char)greens[0];
            bluredData[i][j + 1].green = (unsigned char)greens[1];
            bluredData[i][j + 2].green = (unsigned char)greens[2];
            bluredData[i][j + 3].green = (unsigned char)greens[3];
            bluredData[i][j].blue = (unsigned char)blues[0];
            bluredData[i][j + 1].blue = (unsigned char)blues[1];
            bluredData[i][j + 2].blue = (unsigned char)blues[2];
            bluredData[i][j + 3].blue = (unsigned char)blues[3];
        }
    }

    return bluredData;
}

Pixel** gaussianFilterCl(Pixel** image, int width, int height) {
    double* gaussKern = new double[kernelSize * kernelSize];
    int k = 0;

    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            gaussKern[k++] = kernel[i][j];
        }
    }

    const int kernelRadius = kernelSize / 2;
    const char fileName[] = "Kernel_Gauss.cl";
    int args[] = { kernelSize, kernelRadius, width, height };
    char* kernelSource;
    int size = width * height * 3;
    unsigned char* rawData = unparseData(image, width, height);
    unsigned char* rawBluredData = new unsigned char[size];
    size_t kernelSize;
    FILE* fp;
    
    cl_int err;
    cl_uint errNumPlatforms, errNumDevices;
    cl_platform_id platformID;
    cl_device_id deviceID;
    cl_context context;
    cl_command_queue commandQueue;
    cl_mem rawDataBuffer, rawDataBluredBuffer, argsBuffer, gaussBuffer;
    cl_program program;
    cl_kernel kernel;

    err = clGetPlatformIDs(1, &platformID, &errNumPlatforms);
    err = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, &errNumDevices);
    context = clCreateContext(NULL, errNumDevices, &deviceID, NULL, NULL, &err);
    commandQueue = clCreateCommandQueueWithProperties(context, deviceID, NULL, &err);

    rawDataBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * size, NULL, &err);
    rawDataBluredBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * size, NULL, &err);
    gaussBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * args[0] * args[0], NULL, &err);
    argsBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * 4, NULL, &err);

    fp = fopen(fileName, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    kernelSource = (char*)malloc(MAX_SOURCE_SIZE);
    kernelSize = fread(kernelSource, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, (const size_t*)&kernelSize, &err);
    err = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "gaussianFilter", &err);

    err = clEnqueueWriteBuffer(commandQueue, rawDataBuffer, CL_TRUE, 0, sizeof(unsigned char) * size, rawData, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(commandQueue, rawDataBluredBuffer, CL_TRUE, 0, sizeof(unsigned char) * size, rawBluredData, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(commandQueue, gaussBuffer, CL_TRUE, 0, sizeof(double) * args[0] * args[0], gaussKern, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(commandQueue, argsBuffer, CL_TRUE, 0, sizeof(int) * 4, args, 0, NULL, NULL);


    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &rawDataBuffer);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &rawDataBluredBuffer);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &gaussBuffer);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &argsBuffer);
    

    size_t globalWorkSize[1] = { size };

    auto start = std::chrono::steady_clock::now();

    err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    err = clEnqueueReadBuffer(commandQueue, rawDataBluredBuffer, CL_TRUE, 0, sizeof(unsigned char) * size, rawBluredData, 0, NULL, NULL);

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << static_cast<double>(elapsed) / 1000000 << " s" << std::endl;

    return parseData(rawBluredData, width, height);
}

Pixel** useNegative(Pixel** parsedData, int width, int height) {
    auto start = std::chrono::steady_clock::now();

    Pixel** negativeData = negativeFilter(parsedData, width, height);

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << std::endl << "Negative filter (consequentially) elapsed: " << static_cast<double>(elapsed) / 1000000 << " s" << std::endl;

    return negativeData;
}

Pixel** useNegativeVec(Pixel** parsedData, int width, int height) {
    auto start = std::chrono::steady_clock::now();

    Pixel** negativeData = negativeFilterVec(parsedData, width, height);

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "Negative filter (vectorized) elapsed: " << static_cast<double>(elapsed) / 1000000 << " s" << std::endl;

    return negativeData;
}

Pixel** useNegativeCl(Pixel** parsedData, int width, int height) {
    auto start = std::chrono::steady_clock::now();
    std::cout << "Negative filter (openCl) elapsed: ";

    Pixel** negativeData = negativeFilterCl(parsedData, width, height);

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return negativeData;
}

Pixel** useGauss(Pixel** parsedData, int width, int height) {
    auto start = std::chrono::steady_clock::now();

    Pixel** bluredData = gaussianFilter(parsedData, width, height);

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << std::endl <<"Gauss filter (consequentially) elapsed: " << static_cast<double>(elapsed) / 1000000 << " s" << std::endl;

    return bluredData;
}

Pixel** useGaussVec(Pixel** parsedData, int width, int height) {
    auto start = std::chrono::steady_clock::now();

    Pixel** bluredData = gaussianFilterVec(parsedData, width, height);

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "Gauss filter (vectorized) elapsed: " << static_cast<double>(elapsed) / 1000000 << " s" << std::endl;

    return bluredData;
}

Pixel** useGaussCl(Pixel** parsedData, int width, int height) {
    auto start = std::chrono::steady_clock::now();
    std::cout << "Gauss filter (openCl) elapsed: ";

    Pixel** negativeData = gaussianFilterCl(parsedData, width, height);

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return negativeData;
}

int main() {
    const char* filename = "2400x2400.png";
    const char* negativeFile = "negative.png";
    const char* bluredFile = "blured.png";

    int width, height, numChannels;
    unsigned char* image_data = stbi_load(filename, &width, &height, &numChannels, 0);
    if (!image_data) {
        std::cerr << "Error loading image\n";
        return 1;
    }
    if (numChannels != 3) {
        std::cerr << "Error: onle 3 channels supported!\n";
        
        return 1;
    }
    
    std::cout << "Opened " << filename << std::endl;

    Pixel** parsedData = parseData(image_data, width, height);

    useNegative(parsedData, width, height);
    Pixel** negativeData = useNegativeVec(parsedData, width, height);
    useNegativeCl(parsedData, width, height);

    GenerateGaussianKernel();

    useGauss(parsedData, width, height);
    Pixel** bluredData = useGaussVec(parsedData, width, height);
    useGaussCl(parsedData, width, height);
   
    if (!stbi_write_png(negativeFile, width, height, numChannels,
        unparseData(negativeData, width, height), width * numChannels)) {
        std::cerr << "Error saving negative image\n";
        return 1;
    }
    std::cout << std::endl << "Negative image saved to " << negativeFile << std::endl;

    if (!stbi_write_png(bluredFile, width, height, numChannels,
        unparseData(bluredData, width, height), width * numChannels)) {
        std::cerr << "Error saving blured image\n";
        return 1;
    }
    std::cout << "Blured image saved to " << bluredFile << std::endl;

    stbi_image_free(image_data);

    return 0;
}