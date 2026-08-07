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

## Experiment E1

### Title
Random Horizontal Flip

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