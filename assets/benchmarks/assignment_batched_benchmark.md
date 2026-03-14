# Batched Assignment Benchmark

## Total batch latency (ms)

| Config |unitrack_lapjvx_loop | unitrack_lapjvs_loop | unitrack_lapjvx_batch | unitrack_lapjvs_batch | hungarian_loop_cpu |
|---|---|---|---|---|---|
| iou_8x8_B1 | 0.177 ± 0.012 | 0.162 ± 0.004 | 0.208 ± 0.022 | 0.162 ± 0.004 | 0.071 ± 0.006 |
| iou_8x8_B8 | 1.270 ± 0.065 | 1.291 ± 0.051 | 1.357 ± 0.047 | 1.376 ± 0.097 | 0.526 ± 0.006 |
| iou_8x12_B8 | 1.229 ± 0.016 | 1.480 ± 0.128 | 1.264 ± 0.039 | 1.458 ± 0.121 | 0.575 ± 0.061 |
| iou_16x16_B1 | 0.160 ± 0.005 | 0.163 ± 0.004 | 0.198 ± 0.040 | 0.162 ± 0.006 | 0.136 ± 0.005 |
| iou_16x16_B8 | 1.322 ± 0.143 | 1.346 ± 0.078 | 1.275 ± 0.044 | 1.273 ± 0.047 | 1.137 ± 0.039 |
| iou_16x24_B8 | 1.312 ± 0.070 | 1.562 ± 0.219 | 1.293 ± 0.035 | 1.408 ± 0.179 | 1.137 ± 0.056 |
| iou_32x32_B1 | 0.163 ± 0.005 | 0.167 ± 0.005 | 0.166 ± 0.004 | 0.170 ± 0.009 | 0.153 ± 0.031 |
| iou_32x32_B8 | 1.302 ± 0.010 | 1.327 ± 0.012 | 1.351 ± 0.107 | 1.493 ± 0.164 | 1.245 ± 0.029 |
| iou_32x48_B8 | 1.411 ± 0.113 | 1.405 ± 0.015 | 1.669 ± 0.271 | 1.467 ± 0.144 | 1.236 ± 0.111 |
| iou_64x64_B1 | 0.335 ± 0.070 | 0.302 ± 0.009 | 0.374 ± 0.034 | 0.304 ± 0.009 | 0.218 ± 0.009 |
| iou_64x64_B8 | 1.755 ± 0.012 | 1.800 ± 0.046 | 1.848 ± 0.114 | 1.792 ± 0.085 | 1.926 ± 0.025 |
| iou_64x96_B8 | 1.725 ± 0.096 | 1.723 ± 0.051 | 1.724 ± 0.122 | 1.726 ± 0.112 | 1.624 ± 0.117 |
| iou_128x128_B1 | 0.320 ± 0.005 | 0.321 ± 0.007 | 0.358 ± 0.073 | 0.320 ± 0.006 | 0.622 ± 0.051 |
| iou_128x128_B4 | 1.439 ± 0.067 | 1.434 ± 0.048 | 1.515 ± 0.136 | 1.443 ± 0.054 | 2.345 ± 0.079 |
| iou_128x192_B4 | 1.419 ± 0.031 | 1.419 ± 0.014 | 1.673 ± 0.124 | 1.647 ± 0.165 | 1.755 ± 0.049 |
| iou_256x256_B1 | 1.414 ± 0.028 | 1.408 ± 0.011 | 1.151 ± 0.074 | 1.120 ± 0.113 | 2.030 ± 0.067 |
| iou_256x256_B4 | 7.305 ± 0.055 | 7.375 ± 0.082 | 5.825 ± 0.393 | 5.777 ± 0.466 | 8.547 ± 0.294 |
| gated_32x32_B4 | 0.817 ± 0.066 | 0.714 ± 0.056 | 0.808 ± 0.137 | 0.670 ± 0.006 | 0.619 ± 0.048 |
| gated_64x64_B4 | 0.824 ± 0.009 | 0.835 ± 0.012 | 0.844 ± 0.070 | 0.856 ± 0.050 | 0.869 ± 0.011 |
| gated_128x128_B4 | 1.548 ± 0.008 | 1.656 ± 0.175 | 1.660 ± 0.128 | 1.572 ± 0.044 | 2.161 ± 0.057 |

## Amortized per-problem latency (µs)

| Config |unitrack_lapjvx_loop | unitrack_lapjvs_loop | unitrack_lapjvx_batch | unitrack_lapjvs_batch | hungarian_loop_cpu |
|---|---|---|---|---|---|
| iou_8x8_B1 | 172.4 | 160.8 | 203.7 | 159.9 | 67.3 |
| iou_8x8_B8 | 156.0 | 158.6 | 169.6 | 169.1 | 65.8 |
| iou_8x12_B8 | 153.3 | 189.9 | 155.1 | 185.3 | 68.5 |
| iou_16x16_B1 | 157.2 | 162.1 | 173.5 | 160.3 | 135.1 |
| iou_16x16_B8 | 158.3 | 167.7 | 157.2 | 156.2 | 140.3 |
| iou_16x24_B8 | 160.3 | 178.7 | 159.8 | 162.0 | 141.1 |
| iou_32x32_B1 | 160.2 | 163.8 | 164.7 | 164.8 | 140.0 |
| iou_32x32_B8 | 162.2 | 165.3 | 163.6 | 182.7 | 153.9 |
| iou_32x48_B8 | 171.6 | 174.9 | 216.4 | 175.8 | 149.9 |
| iou_64x64_B1 | 300.3 | 297.2 | 361.0 | 303.0 | 216.8 |
| iou_64x64_B8 | 218.8 | 223.6 | 222.5 | 218.8 | 239.3 |
| iou_64x96_B8 | 211.3 | 213.0 | 209.9 | 209.5 | 195.8 |
| iou_128x128_B1 | 317.3 | 318.1 | 322.1 | 319.3 | 609.9 |
| iou_128x128_B4 | 353.1 | 354.6 | 367.4 | 355.1 | 575.2 |
| iou_128x192_B4 | 352.3 | 354.3 | 414.8 | 406.8 | 435.1 |
| iou_256x256_B1 | 1399.6 | 1407.0 | 1118.4 | 1058.7 | 1985.9 |
| iou_256x256_B4 | 1822.5 | 1838.4 | 1438.9 | 1394.1 | 2092.0 |
| gated_32x32_B4 | 203.8 | 170.2 | 197.3 | 167.0 | 150.4 |
| gated_64x64_B4 | 206.2 | 208.6 | 204.9 | 208.2 | 216.6 |
| gated_128x128_B4 | 386.8 | 394.2 | 397.5 | 389.2 | 534.5 |

## Solution Quality (cost ratio vs. SciPy)

| Config |unitrack_lapjvx_loop | unitrack_lapjvs_loop | unitrack_lapjvx_batch | unitrack_lapjvs_batch | hungarian_loop_cpu |
|---|---|---|---|---|---|
| iou_8x8_B1 | optimal | optimal | optimal | optimal | optimal |
| iou_8x8_B8 | optimal | optimal | optimal | optimal | optimal |
| iou_8x12_B8 | optimal | optimal | optimal | optimal | optimal |
| iou_16x16_B1 | optimal | optimal | optimal | optimal | optimal |
| iou_16x16_B8 | optimal | optimal | optimal | optimal | optimal |
| iou_16x24_B8 | optimal | optimal | optimal | optimal | optimal |
| iou_32x32_B1 | optimal | optimal | optimal | optimal | optimal |
| iou_32x32_B8 | optimal | optimal | optimal | optimal | optimal |
| iou_32x48_B8 | optimal | optimal | optimal | optimal | optimal |
| iou_64x64_B1 | optimal | optimal | optimal | optimal | optimal |
| iou_64x64_B8 | optimal | optimal | optimal | optimal | optimal |
| iou_64x96_B8 | optimal | optimal | optimal | optimal | optimal |
| iou_128x128_B1 | optimal | optimal | optimal | optimal | optimal |
| iou_128x128_B4 | optimal | optimal | optimal | optimal | optimal |
| iou_128x192_B4 | optimal | optimal | optimal | optimal | optimal |
| iou_256x256_B1 | optimal | optimal | optimal | optimal | optimal |
| iou_256x256_B4 | optimal | optimal | optimal | optimal | optimal |
| gated_32x32_B4 | optimal | optimal | optimal | optimal | optimal |
| gated_64x64_B4 | optimal | optimal | optimal | optimal | optimal |
| gated_128x128_B4 | optimal | optimal | optimal | optimal | optimal |
