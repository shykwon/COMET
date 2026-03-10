# Solar Pilot 실험 메모

## 2026-03-09: 초기 sanity check

### COMET 기본 (seed=42)
- log: `comet_solar_K16_s42_20260309_145454`
- ObsMAE=2.122, MissMAE=2.430
- Early stop epoch 55 (best=35), patience=20
- Stage 전환: 1→2 (epoch 11), 2→3 (epoch 15)
- Perplexity 변화: ~14 (S1) → ~5 (S2) → ~13 (S3)
- batch_size=64는 초기 run에서 불안정했으나 다른 버그가 원인일 수 있음
- 다음 실험부터 bs=64 재시도, 불안정 시 32로 타협

### ts_input ablation (seed=42) — 완료
- log: `comet_solar_K16_s42_20260309_222200`
- ObsMAE=2.600, MissMAE=2.675
- **COMET 기본 대비 ObsMAE +22.5% 악화** (2.122 → 2.600)
- patch embedding을 유지하는 기본 모드가 ts_input보다 확실히 우수

### 폐기된 run
- `comet_solar_K16_s42_20260309_144910` — bs=64, 불안정으로 중단
- `comet_solar_K16_s42_20260309_145030` — 시작 실패

### TODO
- [x] ts_input 완료 — ObsMAE=2.600 (기본 2.122 대비 +22.5%)
- [ ] 10-seed 본실험 전 나머지 ablation pilot (no_codebook, temporal variants)
- [ ] traffic 데이터셋은 batch_size=8 필요할 수 있음 (862 variates)
