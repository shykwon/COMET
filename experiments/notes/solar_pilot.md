# Solar Pilot 실험 메모

## 2026-03-09: 초기 sanity check

### COMET 기본 (seed=42)
- log: `comet_solar_K16_s42_20260309_145454`
- ObsMAE=2.122, MissMAE=2.430
- Early stop epoch 55 (best=35), patience=20
- Stage 전환: 1→2 (epoch 11), 2→3 (epoch 15)
- Perplexity 변화: ~14 (S1) → ~5 (S2) → ~13 (S3)
- batch_size=64는 학습 불안정 → **16으로 고정**

### ts_input ablation (seed=42) — 진행중
- log: `comet_solar_K16_s42_20260309_222200` ← 활성
- log: `comet_solar_K16_s42_20260309_222210` ← 중복, 무시
- 두 config 동일 (ts_input=true, 나머지 기본값)
- 현재 epoch 33/150, Stage 3, obs=3.020
- ablation_ts_input.log = ablation_all.log (같은 파일)

### 폐기된 run
- `comet_solar_K16_s42_20260309_144910` — bs=64, 불안정으로 중단
- `comet_solar_K16_s42_20260309_145030` — 시작 실패

### TODO
- [ ] ts_input 완료 후 결과 pilot_solar.json에 기록
- [ ] 10-seed 본실험 전 나머지 ablation pilot (no_codebook, temporal variants)
- [ ] traffic 데이터셋은 batch_size=8 필요할 수 있음 (862 variates)
