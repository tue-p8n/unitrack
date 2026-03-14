# HOTA benchmark — cityscapes-dvps

- **dataset**: cityscapes-dvps
- **device**: cpu
- **models**: {'mask2former-tiny': 'facebook/mask2former-swin-tiny-cityscapes-panoptic', 'mask2former-small': 'facebook/mask2former-swin-small-cityscapes-panoptic', 'mask2former-base': 'facebook/mask2former-swin-base-IN21k-cityscapes-panoptic', 'mask2former-large': 'facebook/mask2former-swin-large-cityscapes-panoptic'}
- **limit_seqs**: 5
- **max_frames**: None
- **mask_iou_threshold**: 0.5
- **min_score**: 0.1
- **offset**: 1000
- **thing_ids**: [11, 12, 13, 14, 15, 16, 17, 18]
- **unitrack_version**: 2.0.0
- **transformers_version**: 5.9.0

| Model | HOTA | DetA | AssA | LocA | MOTA | IDF1 | frames | sec |
|---|---|---|---|---|---|---|---|---|
| mask2former-tiny | 0.3386 | 0.3027 | 0.4322 | 0.7753 | -0.1210 | 0.2556 | 30 | 185.9 |
| mask2former-small | 0.3496 | 0.3147 | 0.4487 | 0.7744 | -0.0813 | 0.2720 | 30 | 197.5 |
| mask2former-base | 0.3528 | 0.3142 | 0.4476 | 0.7793 | -0.0714 | 0.2781 | 30 | 342.7 |
| mask2former-large | 0.3513 | 0.3211 | 0.4363 | 0.7843 | -0.0774 | 0.2800 | 30 | 156.7 |
