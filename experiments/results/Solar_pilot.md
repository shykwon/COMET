# Solar 실험 결과 (Pilot, seed 0)

- 조건: K=16, bs=64, seq_len=12, pred_len=12, missing_rate=0.85
- 프로토콜: seed 0 × 100-mask (10-seed 전체 실험은 다른 서버에서 진행)
- 아키텍처: 1안 (codebook gating)

## Gating vs nocb (seed 0, 100-mask)

| Model | ObsMAE | ObsRMSE |
|-------|--------|---------|
| COMET (gating) | 2.10(0.18) | 3.60(0.40) |
| nocb | 2.43(0.22) | 4.12(0.49) |
| Δ | **-13.9%** | **-12.8%** |

## ts_input ablation (seed 0, 100-mask)

| Model | ObsMAE | ObsRMSE | ppl |
|-------|--------|---------|-----|
| 임베딩 입력 (default) | 2.10(0.18) | 3.60(0.40) | 16.0 |
| 시계열 입력 (ts_input) | 3.07(0.53) | 5.18(1.02) | 15.7 |
| Δ | **+46%** | **+44%** | - |

## VIDA 비교 참고 (MTGNN)

| Setting | MAE | RMSE |
|---------|-----|------|
| Partial | 4.36(0.53) | 6.04(0.81) |
| Oracle | 2.94(0.27) | 4.66(0.57) |
| VIDA | **2.33(0.25)** | **3.71(0.50)** |
| COMET (seed 0) | 2.10(0.18) | 3.60(0.40) |

## 분석

- Solar(137변수)에서 codebook gating 효과 명확 (-13.9%)
- COMET이 VIDA(2.33) 대비 개선 (2.10), Oracle(2.94)도 크게 하회
- ts_input은 임베딩 입력 대비 46% 나쁨 → 임베딩 수준 입력의 구조적 이점 확인
- ts_input에서도 codebook 붕괴 없음 (ppl=15.7) — ECG5000과 대조적
- 10-seed 전체 결과는 다른 서버 실험 완료 후 업데이트 필요
