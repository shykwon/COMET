# Solar Pilot 실험 메모

## 2026-03-09: 초기 sanity check

### COMET 기본 (seed=42)
- ObsMAE=2.122, MissMAE=2.430
- Early stop epoch 55 (best=35), patience=20
- Stage 전환: 1→2 (epoch 11), 2→3 (epoch 15)
- Perplexity 변화: ~14 (S1) → ~5 (S2) → ~13 (S3)
- batch_size=64는 학습 불안정 → **16으로 고정**

### ts_input ablation (seed=42)
- 진행중 (222200, 222210)
- 222210은 중복 run — 하나만 사용

### TODO
- [ ] 10-seed 본실험 시작 전 나머지 ablation pilot도 돌려볼 것
- [ ] traffic 데이터셋은 batch_size=8 필요할 수 있음 (862 variates)
