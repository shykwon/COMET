# ETTh1 실험 결과 (Gating)

- 조건: K=8, bs=64, seq_len=12, pred_len=12, missing_rate=0.85
- 아키텍처: 1안 (codebook gating)
- RMSE: per-horizon 평균 (Chauhan/VIDA 호환)
- 프로토콜: 10-seed × 100-mask = 1,000 runs

## MTGNN COMET (VIDA 형식)

| Model | ObsMAE | ObsRMSE | runs |
|-------|--------|---------|------|
| COMET (gating) | 1.66(0.87) | 2.90(1.70) | 1000 |

## 참고: 이전 원본 아키텍처 결과 (seed간 std)

| Model | ObsMAE | ObsRMSE |
|-------|--------|---------|
| mtgnn comet (원본) | 1.6528±0.0427 | 2.8843±0.0659 |
| mtgnn nocb (원본) | 1.6533±0.0193 | 2.9190±0.0373 |
| mstgcn comet | 1.6555±0.0294 | - |
| mstgcn nocb | 1.7227±0.0275 | - |
| astgcn comet | 1.7047±0.0340 | - |
| astgcn nocb | 1.7411±0.0731 | - |
| tgcn comet | 1.7055±0.0569 | - |
| tgcn nocb | 1.7483±0.0280 | - |

## 분석

- ETTh1(7변수)에서는 codebook 효과 미미 (변수 수 부족)
- gating(1.66) vs 원본(1.6528) 거의 동일
- mstgcn에서 codebook 효과가 가장 명확 (-3.9%)
- 이전 결과는 원본 아키텍처(gating 없음) + decoder 스킵 nocb + flat RMSE
