# Experiment Log

## Baseline (v1.0)

Model:
- Custom CNN

Result:
- Train Accuracy: 93.36%
- Validation Accuracy: 86.96%
- Test Accuracy: 85.93%

Status:
✅ Accepted as Baseline

## Experiment E1 - Random Horizontal Flip

### Objective
Evaluate whether horizontal flipping improves generalization.

### Change
Added:
- RandomHorizontalFlip(p=0.5)

### Results

Train Accuracy: 84.37
Validation Accuracy: 85.86
Test Accuracy: 85.13%

### Baseline

Test Accuracy: 85.93%

### Conclusion

❌ Rejected

Horizontal flipping reduced test accuracy by approximately 0.8 percentage points compared to the baseline.

## E2 — Learning Rate Tuning

### Objective

Evaluate whether reducing the initial learning rate improves model generalization.

### Change

Baseline:

- Learning rate: `0.001`

E2:

- Learning rate: `0.0005`

No other model or training configuration was changed.

### Results

| Metric | v1.0 Baseline | E2 |
|---|---:|---:|
| Train Accuracy | 93.36% | 90.47% |
| Validation Accuracy | 86.96% | 84.97% |
| Test Accuracy | 85.93% | **86.00%** |

### Conclusion

E2 produced a very small improvement in test accuracy:

`85.93% → 86.00%` (**+0.07 percentage points**).

However, validation accuracy decreased from `86.96%` to `84.97%`, so the improvement is too small to consider the learning-rate change a meaningful performance improvement.

**Decision: Keep the experiment result recorded, but use the baseline learning rate `0.001` for the next experiment.**

## E3 — Weight Decay

### Objective

Evaluate whether weight decay can reduce overfitting and improve generalization.

### Change

Baseline:

- Learning rate: `0.001`
- Weight decay: `0`

E3:

- Learning rate: `0.001`
- Weight decay: `0.0001`

No other model or training configuration was changed.

### Results

| Metric | v1.0 Baseline | E3 |
|---|---:|---:|
| Train Accuracy | 93.36% | 84.16% |
| Validation Accuracy | 86.96% | 84.61% |
| Test Accuracy | 85.93% | 84.60% |

### Conclusion

Weight decay did not improve performance.

Test accuracy decreased from `85.93%` to `84.60%`
(**−1.33 percentage points**).

The experiment was therefore rejected.

**Decision: Do not use weight decay for the next experiment.**

## E4 — CNN Architecture Improvement

### Objective

Evaluate whether increasing the CNN's feature extraction capacity improves image classification performance.

### Change

Baseline:

- Convolution blocks: `3`
- Channels: `3 → 32 → 64 → 128`

E4:

- Convolution blocks: `4`
- Channels: `3 → 32 → 64 → 128 → 256`

No other model or training configuration was changed.

### Results

| Metric              | v1.0 Baseline |     E4 |
| ------------------- | ------------: | -----: |
| Train Accuracy      |        93.36% | 91.12% |
| Validation Accuracy |        86.96% | 87.10% |
| Test Accuracy       |        85.93% | **87.77%** |

### Conclusion

Adding a fourth convolution block improved the model's test performance.

Test accuracy increased from `85.93%` to `87.77%`
(**+1.84 percentage points**).

The additional convolution block provided greater feature extraction capacity and improved generalization.

**Decision: Keep E4 as the new baseline.**