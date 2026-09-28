# MXFP4 × MXFP4 GEMM

V1 had roughly 0.60x geomean speedup, 

V2, warp specialization , single CTA tensor core MMA
Shape                 Kernel  Kernel TFLOPS  FlashInfer/CuTe MXFP4  Baseline TFLOPS  Speedup
-----------------  ---------  -------------  ---------------------  ---------------  -------
M128 N128 K128       8.19 us            0.5                8.08 us              0.5    0.99x
M256 N256 K256       8.35 us            4.0                8.21 us              4.1    0.98x
M512 N512 K512       8.22 us           32.6                8.19 us             32.8    1.00x
M1024 N1024 K1024   10.24 us          209.7                8.19 us            262.1    0.80x
M2048 N2048 K2048   14.34 us        1,198.4               12.26 us          1,401.8    0.85x
M4096 N4096 K4096   40.96 us        3,355.4               32.80 us          4,190.2    0.80x
M8192 N8192 K8192  206.85 us        5,315.6              159.74 us          6,883.0    0.77x
-----------------  ---------  -------------  ---------------------  ---------------  -------
Geomean                                                                                0.88x


V3 has 0.83 geomean speedup