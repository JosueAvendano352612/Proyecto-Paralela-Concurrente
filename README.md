# Optimización y Paralelización de Algoritmos Fractales (Mandelbrot) y Convolución 2D

Este repositorio contiene el código fuente del proyecto final de Programación Paralela y Concurrente. El objetivo del proyecto es optimizar la generación de una imagen 8K del Conjunto de Mandelbrot y la aplicación de un filtro de desenfoque Gaussiano, partiendo de un código secuencial hasta llegar a una implementación altamente optimizada utilizando **OpenMP** en C++.

## Optimizaciones implementadas:
* Paralelización básica con OpenMP.
* Balanceo de carga mediante Schedulers (`dynamic`).
* Mitigación de Falsos Compartimientos (*False Sharing*) en el cálculo del histograma usando memoria local.
* Vectorización forzada SIMD (Estructura SPMD).

**Hardware de prueba:** AMD Ryzen 5 5600 (6 Núcleos / 12 Hilos).
