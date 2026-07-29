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

