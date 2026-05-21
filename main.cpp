#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

const int WIDTH = 7680;
const int HEIGHT = 4320;
const int MAX_ITER = 256;

void generateMandelbrot(std::vector<int>& image) {
    // 1. SCHEDULERS: Aquí está en "dynamic". Para tu reporte, cambia esto a "static" o "guided" 
    // y evalúa el tiempo para ver cuál es el óptimo para tu Ryzen.
    #pragma omp parallel for schedule(dynamic, 64)
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

void applyBlurAndHistogram(const std::vector<int>& input, std::vector<int>& output, std::vector<int>& histogram) {
    int kernel[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };
    
    #pragma omp parallel
    {
        // 2. FALSOS COMPARTIMIENTOS: Se usa un arreglo estrictamente local por hilo 
        // para evitar colisiones en la memoria caché (False Sharing).
        std::vector<int> local_hist(MAX_ITER + 1, 0);

        #pragma omp for schedule(static)
        for (int y = 1; y < HEIGHT - 1; ++y) {
            
            // 3. SPMD: Forzamos la vectorización del bucle más interno
            #pragma omp simd
            for (int x = 1; x < WIDTH - 1; ++x) {
                int sum = 0;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        sum += input[(y + ky) * WIDTH + (x + kx)] * kernel[ky + 1][kx + 1];
                    }
                }
                int final_val = sum / 16;
                output[y * WIDTH + x] = final_val;
                local_hist[final_val]++; 
            }
        }

        // Sincronización final: Sumamos los arreglos locales al global de forma segura
        #pragma omp critical
        {
            for (int i = 0; i <= MAX_ITER; ++i) {
                histogram[i] += local_hist[i];
            }
        }
    }
}

int main() {
    std::vector<int> image(WIDTH * HEIGHT);
    std::vector<int> blurredImage(WIDTH * HEIGHT);
    std::vector<int> histogram(MAX_ITER + 1, 0);

    auto start = std::chrono::high_resolution_clock::now();
    
    generateMandelbrot(image);
    applyBlurAndHistogram(image, blurredImage, histogram);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Tiempo optimizado (Schedulers, SPMD, Histograma): " << diff.count() << " s\n";

    return 0;
}