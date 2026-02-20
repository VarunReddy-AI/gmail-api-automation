# Model Training Metrics

**Run Time:** 2026-02-18 12:31:35.610911

**Model Path:** artifacts/model.pkl

## Test Metrics
- Accuracy: 0.9168

### Confusion Matrix
```
[[ 21   0   0   0   0]
 [  1 224   4  34   0]
 [  0   0 176   3   0]
 [  2  22   7 416   0]
 [  0   1   1   5  44]]
```

### Classification Report
```
              precision    recall  f1-score   support

           0       0.88      1.00      0.93        21
           1       0.91      0.85      0.88       263
           2       0.94      0.98      0.96       179
           3       0.91      0.93      0.92       447
           4       1.00      0.86      0.93        51

    accuracy                           0.92       961
   macro avg       0.93      0.93      0.92       961
weighted avg       0.92      0.92      0.92       961

```

## Train Metrics
- Accuracy: 0.9622

### Confusion Matrix
```
[[  64    0    0    0    0]
 [   2  757    8   18    3]
 [   1    1  529    5    0]
 [   4   36   17 1270   13]
 [   0    0    0    1  154]]
```

### Classification Report
```
              precision    recall  f1-score   support

           0       0.90      1.00      0.95        64
           1       0.95      0.96      0.96       788
           2       0.95      0.99      0.97       536
           3       0.98      0.95      0.96      1340
           4       0.91      0.99      0.95       155

    accuracy                           0.96      2883
   macro avg       0.94      0.98      0.96      2883
weighted avg       0.96      0.96      0.96      2883

```
