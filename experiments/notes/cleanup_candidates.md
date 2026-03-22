# 최종 아키텍처 확정 후 삭제/정리 대상

> hard_nograd 실험 결과 확정 후 최종 판단. 현재는 마킹만.

## 코드 삭제 대상

### codebook.py
- [ ] `soft_lookup()` — hard로 확정되면 제거
- [ ] `hard_lookup()` (ST 버전) — hard_nograd로 확정되면 제거
- [ ] `tau` 파라미터 — hard에서 forward 시 미사용 (ST backward용이었음)
- [ ] `usage_ema` 버퍼 — revival 제거 시 불필요 (단, EMA update의 활용도 추적용이면 유지)

### decoder.py
- [ ] `use_film` 관련 코드 — FiLM 채택 안 하면 제거 (film_gamma, film_beta, film_norm)
- [ ] `use_direct_add` 관련 코드 — ablation 완료 후 제거
- [ ] `cross_attn_obs_refine` — direct_add/film 미채택 시 유지 (baseline은 cross-attn)

### comet.py
- [ ] `hard_lookup` 플래그 — hard_nograd로 통합되면 제거
- [ ] `use_film` 플래그 — 미채택 시 제거
- [ ] `use_direct_add` 플래그 — 미채택 시 제거

### train.py CLI
- [ ] `--hard_lookup` — hard_nograd가 기본이 되면 제거
- [ ] `--hard_nograd` — 기본 동작이 되면 플래그 자체 불필요
- [ ] `--film` — 미채택 시 제거
- [ ] `--direct_add` — 미채택 시 제거
- [ ] `--no_revival` — revival 제거 확정 시 제거
- [ ] `--no_ema` — EMA가 기본이면 disable_ema config로 충분
- [ ] `--entropy_reg_weight 0` — entropy 제거 확정 시 default.yaml에서 제거

### configs/default.yaml
- [ ] `tau: 0.5` — hard에서 불필요
- [ ] `entropy_reg_weight: 0.2` — 제거 확정 시 삭제
- [ ] `disable_ema` 키 자체 — EMA가 항상 ON이면 키 불필요

### losses.py
- [ ] `compute_entropy_reg()` — entropy 제거 확정 시

### curriculum.py
- [ ] entropy 관련 로직 (있다면)

## 코드 변경 대상 (삭제가 아닌 수정)

### codebook.py
- [ ] `hard_lookup_nograd()` → `lookup()`으로 이름 변경 (유일한 lookup이 되므로)
- [ ] `ema_update()` — no_revival 파라미터 제거, revival 코드 제거 (revival 불필요 확정 시)

### comet.py
- [ ] forward에서 3-way lookup 분기 → 단일 호출로 단순화
- [ ] `__init__` 파라미터 정리 (hard_lookup, hard_nograd, use_film, use_direct_add 제거)

### train.py
- [ ] ablation 태그 로직 (_hardlk, _hardng, _norev, _noema, _noent, _film, _dadd) 제거
- [ ] 디렉토리 네이밍 단순화

## 유지해야 할 것

- [ ] `soft_lookup` — evaluate.py에서 forward_full(teacher)이 사용. teacher도 hard로 바꿀지 결정 필요
- [ ] `cross_attn_cb` (Stage B) — ablation에서 핵심으로 확인됨
- [ ] `cross_attn_obs` (Stage A) — 관측 패치 기반 복원
- [ ] `ema_update` — codebook 학습의 핵심
- [ ] `init_from_kmeans` — 초기화 필수
- [ ] InfoNCE, KL Match loss — encoder 학습
- [ ] 3-Stage Curriculum — 학습 안정성

## 판단 보류

- [ ] `_revive_dead_entries` — entropy 없이 EMA만으로 revival 없이도 ppl 유지되는지 추가 확인 필요
- [ ] `forward_full` (teacher path) — hard_nograd에서 teacher가 여전히 필요한지 (InfoNCE/KL용)
- [ ] KL Match loss — hard lookup에서 w_sub가 one-hot이면 KL 계산이 의미 있는지
