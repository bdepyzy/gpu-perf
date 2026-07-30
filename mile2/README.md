Naive
Suite=core | shapes=3 | warmup=5 | iterations=100 | C[M,N]=A[M,K]@B[K,N]
4096x4096x4096 | CuTe 18845.81 us 7.29 TF/s | torch 347.14 us 395.92 TF/s | 1.84%
4096x14336x4096 | CuTe 66611.38 us 7.22 TF/s | torch 1238.06 us 388.54 TF/s | 1.86%
Stopping app - local entrypoint completed.
4096x4096x14336 | CuTe 66480.88 us 7.24 TF/s | torch 1250.22 us 384.76 TF/s | 1.88%

1 thread per output element, K loop over global


Smem-tiled (no swizzle)
Shapes=7 | warmup=5 | iterations=100 | C[M,N]=A[M,K]@B[K,N]
256x256x256 | CuTe 34.86 us 0.96 TF/s | torch 6.06 us 5.54 TF/s | 17.38%
512x512x512 | CuTe 61.50 us 4.36 TF/s | torch 5.96 us 45.06 TF/s | 9.69%
768x768x768 | CuTe 90.18 us 10.05 TF/s | torch 5.66 us 159.93 TF/s | 6.28%
1024x1024x1024 | CuTe 128.46 us 16.72 TF/s | torch 6.20 us 346.14 TF/s | 4.83%
2048x2048x2048 | CuTe 446.03 us 38.52 TF/s | torch 13.73 us 1251.42 TF/s | 3.08%
4096x4096x4096 | CuTe 3332.78 us 41.24 TF/s | torch 96.27 us 1427.69 TF/s | 2.89%
Stopping app - local entrypoint completed.
8192x8192x8192 | CuTe 29129.32 us 37.75 TF/s | torch 794.24 us 1384.36 TF/s | 2.73%

Smem-tiled (sA Swizzle<2,3,5>, 128-bit tiled-copy staging, LDS.128 b_frag)
Hypothesis: scalar LDS/STS + 2-way ld bank conflicts bound the inner loop. Result: 1.33-1.49x.
Shapes=7 | warmup=5 | iterations=100 | C[M,N]=A[M,K]@B[K,N]
256x256x256 | CuTe 32.80 us 1.02 TF/s | torch 5.66 us 5.93 TF/s | 17.24%
512x512x512 | CuTe 57.36 us 4.68 TF/s | torch 5.79 us 46.35 TF/s | 10.10%
768x768x768 | CuTe 86.01 us 10.53 TF/s | torch 4.96 us 182.50 TF/s | 5.77%
1024x1024x1024 | CuTe 110.74 us 19.39 TF/s | torch 5.75 us 373.20 TF/s | 5.20%
2048x2048x2048 | CuTe 372.21 us 46.16 TF/s | torch 13.59 us 1264.33 TF/s | 3.65%
4096x4096x4096 | CuTe 2508.37 us 54.79 TF/s | torch 94.98 us 1447.11 TF/s | 3.79%
8192x8192x8192 | CuTe 19542.60 us 56.26 TF/s | torch 754.61 us 1457.06 TF/s | 3.86%


Swizzled smem layouts
Shapes=7 | warmup=5 | iterations=100 | C[M,N]=A[M,K]@B[K,N]
256x256x256     | CuTe      32.80 us      1.02 TF/s | torch       9.22 us      3.64 TF/s |  28.10%
512x512x512     | CuTe      57.37 us      4.68 TF/s | torch       9.44 us     28.45 TF/s |  16.45%
768x768x768     | CuTe      85.93 us     10.54 TF/s | torch       8.81 us    102.79 TF/s |  10.26%
1024x1024x1024  | CuTe     110.68 us     19.40 TF/s | torch       9.13 us    235.12 TF/s |   8.25%
2048x2048x2048  | CuTe     371.34 us     46.26 TF/s | torch      13.53 us   1269.98 TF/s |   3.64%
4096x4096x4096  | CuTe    2507.76 us     54.81 TF/s | torch      93.63 us   1467.94 TF/s |   3.73%
8192x8192x8192  | CuTe   19558.60 us     56.22 TF/s | torch     704.05 us   1561.70 TF/s |   3.60%
Stopping app - local entrypoint completed.
✓ App completed. View run at https://modal.com/apps/obo-dan87/main/ap-KqyufeIl2C7A5sl15fi4ee
(learn-kernels) ~/learn/learn-kernels/mile2 git:(main) ✗ »

