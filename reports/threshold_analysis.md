# Threshold analysis (injection class)

Calibrated TF-IDF/SVM — sweep `P(injection)` threshold.

| threshold | accuracy | macro F1 | inj precision | inj recall |
|---:|---:|---:|---:|---:|
| 0.30 | 0.8859 | 0.8796 | 0.8512 | 0.9711 |
| 0.35 | 0.8911 | 0.8859 | 0.8638 | 0.9620 |
| 0.40 | 0.8929 | 0.8885 | 0.8740 | 0.9505 |
| 0.45 | 0.8919 | 0.8880 | 0.8799 | 0.9400 |
| 0.50 | 0.8885 | 0.8850 | 0.8856 | 0.9254 |
| 0.55 | 0.8855 | 0.8827 | 0.8956 | 0.9062 |
| 0.60 | 0.8835 | 0.8813 | 0.9071 | 0.8881 |
| 0.65 | 0.8743 | 0.8726 | 0.9148 | 0.8612 |
| 0.70 | 0.8595 | 0.8583 | 0.9191 | 0.8281 |

- default predict (~0.5): acc=0.8885, macro_f1=0.8850, inj_recall=0.9254
- best macro F1 in sweep: t=0.40 → F1=0.8885
- highest injection recall in sweep: t=0.30 → recall=0.9711

Takeaway: lower the threshold if you want fewer missed jailbreaks (more false alarms).
