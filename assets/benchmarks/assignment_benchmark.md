# Assignment Algorithm Benchmark

## Latency (ms)

| Dataset |Greedy | Hungarian | Auction | Jonker | SoftAssignment |
|---|---|---|---|---|---|
| iou_8x8 | 0.524 ± 0.017 | 0.131 ± 0.009 | 1.340 ± 0.485 | 0.207 ± 0.007 | 11.760 ± 0.873 |
| iou_32x32 | 1.295 ± 0.089 | 0.204 ± 0.005 | 4.509 ± 1.724 | 0.220 ± 0.007 | 12.373 ± 0.429 |
| iou_64x64 | 2.465 ± 0.149 | 0.283 ± 0.013 | 6.824 ± 1.755 | 0.262 ± 0.022 | 15.610 ± 0.709 |
| iou_128x128 | 5.509 ± 0.173 | 0.666 ± 0.048 | 15.050 ± 6.606 | 0.478 ± 0.098 | 22.192 ± 1.640 |
| iou_256x256 | 19.048 ± 0.676 | 2.746 ± 0.221 | 29.538 ± 11.161 | 1.971 ± 0.498 | 37.630 ± 2.559 |
| cdist_64x64 | 2.355 ± 0.101 | 0.301 ± 0.027 | 14.070 ± 8.814 | 0.953 ± 0.505 | 20.184 ± 0.740 |
| cdist_128x128 | 5.472 ± 0.118 | 0.799 ± 0.105 | 33.391 ± 35.753 | 5.922 ± 2.687 | 26.297 ± 0.928 |
| dense_64x16 | 0.791 ± 0.057 | 0.178 ± 0.005 | 0.572 ± 0.104 | 0.229 ± 0.004 | 12.679 ± 0.496 |
| dense_16x64 | 0.787 ± 0.062 | 0.178 ± 0.008 | 0.780 ± 0.165 | 0.229 ± 0.019 | 12.062 ± 0.426 |
| gated_64x64 | 2.289 ± 0.138 | 0.271 ± 0.015 | 95.645 ± 362.374 | 0.271 ± 0.028 | 14.898 ± 0.263 |
| gated_128x128 | 5.336 ± 0.227 | 0.659 ± 0.045 | 241.447 ± 492.326 | 0.523 ± 0.107 | 19.300 ± 0.472 |
| empty_0x0 | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.000 |

## Solution Quality (cost ratio vs. optimal)

| Dataset |Greedy | Hungarian | Auction | Jonker | SoftAssignment |
|---|---|---|---|---|---|
| iou_8x8 | +30.7% | optimal | +8.7% | optimal | optimal |
| iou_32x32 | +92.3% | optimal | +28.3% | optimal | optimal |
| iou_64x64 | +113.4% | optimal | +32.0% | optimal | optimal |
| iou_128x128 | +155.3% | optimal | +43.4% | optimal | optimal |
| iou_256x256 | +196.3% | optimal | +46.8% | optimal | optimal |
| cdist_64x64 | +22.8% | optimal | +18.4% | optimal | optimal |
| cdist_128x128 | +25.8% | optimal | +22.4% | optimal | optimal |
| dense_64x16 | +6.7% | optimal | +277.7% | optimal | +277.1% |
| dense_16x64 | +7.1% | optimal | +0.8% | optimal | +254.9% |
| gated_64x64 | +11.9% | optimal | +30.1% | optimal | optimal |
| gated_128x128 | +43.2% | optimal | +44.2% | optimal | optimal |
| empty_0x0 | optimal | optimal | optimal | optimal | optimal |
