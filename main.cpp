#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <omp.h> // Librería para OpenMP

const int WIDTH = 7680;
const int HEIGHT = 4320;
const int MAX_ITER = 256;

// Tarea A paralela
void generateMandelbrot(std::vector<int>& image) {
    #pragma omp parallel for
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            double zx = 0.0, zy = 0.0;
            double cx = (x - WIDTH / 2.0) * 4.0 / WIDTH;
            double cy = (y - HEIGHT / 2.0) * 4.0 / WIDTH;
            int iter = 0;
            while (zx * zx + zy * zy < 4.0 && iter < MAX_ITER) {
                double tmp = zx * zx - zy * zy + cx;
                zy = 2.0 * zx * zy + cy;
                zx = tmp;
                iter++;
            }
            image[y * WIDTH + x] = iter;
        }
    }
}

// Tarea B paralela
void applyBlur(const std::vector<int>& input, std::vector<int>& output) {
    int kernel[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };
    int kernelWeight = 16;

    #pragma omp parallel for
    for (int y = 1; y < HEIGHT - 1; ++y) {
        for (int x = 1; x < WIDTH - 1; ++x) {
            int sum = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    sum += input[(y + ky) * WIDTH + (x + kx)] * kernel[ky + 1][kx + 1];
                }
            }
            output[y * WIDTH + x] = sum / kernelWeight;
        }
    }
}

int main() {
    std::vector<int> image(WIDTH * HEIGHT);
    std::vector<int> blurredImage(WIDTH * HEIGHT);

    auto start = std::chrono::high_resolution_clock::now();
    
    generateMandelbrot(image);
    applyBlur(image, blurredImage);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Tiempo paralelo (IA base): " << diff.count() << " s\n";

    return 0;
}