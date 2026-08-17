# GPT-only 클릭베이트 제목 선별 구현 기록

작성일: 2026-08-17

## 완료 결과

Stage 3 제목 처리를 `gpt-5.4-mini` 기반의 네 동작으로 구현했다.

1. 최종 길이 검증을 통과한 한국어 대본에서 서로 다른 스타일의 제목 후보 5개를 생성한다.
2. 같은 모델을 별도 구조화 호출로 사용해 후보별 근거성, 클릭베이트 강도, 대중성, safety-ok와 한국어 판정 근거를 받는다.
3. 로컬 코드가 후보 일치 여부, 정확한 대본 근거, 제목 길이, URL, 직접 투자 지시, safety-ok, 근거성 기준을 강제한 뒤 점수순으로 상위 3개를 고른다.
4. 선택 제목, 상위 3개 후보, 각 후보의 심사 점수와 판정 근거를 `scripts.json`에 저장하고 재시작 가능한 상태 DB를 갱신한다.

기본 실행 명령은 다음과 같다.

```powershell
python -m community_shorts generate --model gpt-5.4-mini --title-mode gpt-ranked
```

## 실제 판정 규칙

GPT 심사는 다음 필드를 0~1 범위로 반환한다.

- `evidence_support`: 제목의 핵심 주장을 최종 대본이 직접 뒷받침하는 정도
- `clickbait_strength`: 호기심과 자극을 통해 클릭을 유도하는 정도
- `mass_appeal`: 비기술 대중의 일상, 업무, AI, 테크주 관심과 가까운 정도
- `safety_ok`: 일반 GPT에 입력했을 때 거절·경고·도움 불가 응답을 유발할 수준의 위험이 없는지 여부

로컬 결합 점수는 다음 식으로 계산한다.

```text
0.45 × clickbait_strength
+ 0.35 × mass_appeal
+ 0.20 × evidence_support
```

`safety_ok=false` 또는 `evidence_support<0.80`인 후보는 점수와 무관하게 탈락한다. 통과 후보가 3개 미만이면 임의 후보를 채우지 않고 `title_failed`로 종료한다. “충격”, “무조건”, “드디어 밝혀졌다” 같은 강한 문구 자체는 허용하지만 대본에 없는 사실, 수치, 인과관계, 확실성을 암시하면 근거성 점수를 낮추도록 심사 프롬프트에 명시했다.

## 변경된 런타임 계약

- 제목 후보 스타일에 `curiosity_gap`, `strong_factual_statement`를 추가했다.
- `TitleCandidatePool`은 정확히 5개의 서로 다른 스타일과 제목을 요구한다.
- `TitleJudgeResult`는 생성된 5개 제목과 정확히 일치하는 5개 평가를 요구한다.
- 공개 `TitlePackage`에는 심사를 통과한 서로 다른 3개 스타일을 담는다.
- `GeneratedScript.title_ranking`에는 근거성, 클릭베이트 강도, 대중성, safety-ok, 결합 점수, 한국어 판정 근거를 남긴다.
- CLI의 기본 제목 모드는 `gpt-ranked`이고, 호환성 확인용으로만 `legacy` 모드를 유지한다.

## 취소하고 제거한 접근

초기에는 `jjanoong2/clickbait_embeddings`와 KR-SBERT 임베딩을 이용한 로컬 MLP 학습 경로를 구현하고 50,000페어 학습을 시도했다. 학습 데이터 1.701GB와 검증 데이터 약 424MB를 받은 뒤, 로컬 i5·16GB 환경에서 실행 시간과 데이터 관리 비용이 목적에 비해 크다고 판단해 사용자 지시에 따라 중단했다.

실행 중인 학습 프로세스를 종료했고, 내려받은 `.hf-cache` 약 2.1GB와 미완성 체크포인트를 삭제했다. PyTorch, sentence-transformers, NumPy, scikit-learn, Hugging Face 학습 CLI와 체크포인트 코드는 프로젝트 의존성과 소스에서 모두 제거했다. 외부 저장소의 코드나 모델 가중치는 최종 구현에 포함하지 않았다.

## 검증 결과

구현 중 다음 실패를 재현하고 수정했다.

- Windows 환경에 Python 3.13 가상환경이 없어 새로 구성했다.
- NumPy 객체 배열의 pickle 로딩 차단으로 데이터 읽기가 중단됐으나, 이후 로컬 학습 경로 자체를 폐기했다.
- Stage 3 CLI가 `script_timing`을 전달하지 않아 실패하던 문제를 회귀 테스트로 재현하고 수정했다.
- GPT 심사가 생성 후보를 누락·교체하지 못하도록 후보 집합의 완전 일치를 검증했다.
- safety-ok와 근거성 하드 게이트를 통과한 후보가 3개 미만일 때 결과를 저장하지 않는 테스트를 추가했다.

최종 오프라인 검증은 fixture 기반 전체 Stage 1–3와 OpenAI 구조화 출력 경계를 포함하며 `103 passed, 1 deselected`를 기록했다. 실제 OpenAI 네트워크 스모크 테스트는 기존 로컬 `curated.json` 내용을 외부로 전송하는 별도 승인이 없어 수행하지 않았다.
