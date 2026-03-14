# HOTA benchmark — cityscapes-dvps

- **dataset**: cityscapes-dvps
- **device**: cpu
- **models**: {'mask2former-tiny': 'facebook/mask2former-swin-tiny-cityscapes-panoptic'}
- **limit_seqs**: 5
- **max_frames**: None
- **mask_iou_threshold**: 0.5
- **min_score**: 0.1
- **offset**: 1000
- **thing_ids**: [11, 12, 13, 14, 15, 16, 17, 18]
- **unitrack_version**: 2.0.0
- **transformers_version**: 5.9.0
- **trackers**: ['maskiou', 'cosine', 'cascade', 'kalman', 'learned']

| Model | Tracker | HOTA | DetA | AssA | LocA | MOTA | IDF1 | frames | sec |
|---|---|---|---|---|---|---|---|---|---|
| mask2former-tiny | maskiou | 0.3386 | 0.3027 | 0.4322 | 0.7753 | -0.1210 | 0.2556 | 30 | 676.5 |
| mask2former-tiny | cosine | 0.3741 | 0.2999 | 0.4992 | 0.7755 | 0.0298 | 0.3622 | 30 | 425.4 |
| mask2former-tiny | cascade | 0.4149 | 0.3005 | 0.6121 | 0.7756 | 0.0595 | 0.4133 | 30 | 110.6 |
| mask2former-tiny | kalman | 0.2832 | 0.3033 | 0.2941 | 0.7752 | -0.1964 | 0.1756 | 30 | 95.8 |
| mask2former-tiny | learned | 0.4123 | 0.3011 | 0.6086 | 0.7753 | 0.0397 | 0.3956 | 30 | 89.6 |
