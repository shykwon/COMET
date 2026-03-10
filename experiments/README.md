# experiments/ 정리 규칙

## 디렉토리 구조

```
experiments/
├── README.md                    ← 이 파일 (규칙 정의)
├── plans/
│   └── experiment_matrix.yaml   ← 전체 실험 매트릭스 (단일 파일)
├── results/
│   ├── pilot_<dataset>.json     ← pilot (1-seed sanity check)
│   └── <RQ>_<dataset>.json      ← 본실험 (10-seed × 100-mask)
└── notes/
    └── <dataset>_<topic>.md     ← 분석 메모, 관찰
```

## 파일 네이밍

### results/
- **Pilot**: `pilot_<dataset>.json` — 1-seed sanity check, 논문 미사용
- **본실험**: `RQ1a_solar.json`, `RQ3_ablation_solar.json` 등
- 하나의 RQ + 데이터셋 = 하나의 json

### notes/
- `<dataset>_<topic>.md` — 예: `solar_pilot.md`, `traffic_batch_tuning.md`
- 날짜는 파일 내부 헤딩에 기록 (`## 2026-03-09: ...`)

### plans/
- `experiment_matrix.yaml` 하나로 전체 관리
- 각 run에 `status: TODO | IN_PROGRESS | DONE | FAILED` 표기

## results JSON 포맷

```json
{
  "description": "무엇을 측정하는 실험인지",
  "dataset": "solar",
  "protocol": "10-seed × 100-mask",
  "runs": {
    "<run_name>": {
      "log_dir": "logs/ 내 디렉토리명",
      "seeds": [42, 123, 456, ...],
      "missing_rate": 0.85,
      "config_overrides": { "temporal_type": "transformer" },
      "test": {
        "ObsMAE": { "mean": 2.12, "std": 0.05 },
        "ObsRMSE": { "mean": 3.61, "std": 0.08 },
        "MissMAE": { "mean": 2.43, "std": 0.07 }
      }
    }
  }
}
```

### Pilot (1-seed)일 때는 mean/std 없이 스칼라로:
```json
"test": { "ObsMAE": 2.122, "ObsRMSE": 3.614 }
```

## logs/ vs experiments/ 구분

| 항목 | logs/ (gitignore) | experiments/ (git 추적) |
|------|-------------------|------------------------|
| 체크포인트 (.pt) | O | X |
| train_log.json | O | X |
| config.yaml | O | X |
| 결과 요약 | results.json (원본) | results/<RQ>.json (정제) |
| 분석 메모 | X | notes/ |

**원칙**: logs/는 무겁고 재생성 가능, experiments/는 가볍고 공유 목적

## experiment_matrix.yaml 규칙

- 각 run의 `status` 반드시 유지
- 완료 시 `status: DONE`, 결과 json 경로 추가
- Pilot run은 별도 `pilot:` 섹션에 기록
- 본실험(10-seed)과 pilot(1-seed) 명확히 구분

## 커밋 규칙

- 실험 결과 추가/갱신 시 커밋 메시지: `exp: <내용>`
- 예: `exp: Add RQ3 ablation results for solar`
- push는 `exp/private` 브랜치에서만
