# 🚀 EoMT 모바일 추론 (Mobile Inference) 프로젝트

이 프로젝트는 기존 **EoMT (Encoder only Mask Transformer)** 모델을 Apple의 **Core ML 프레임워크**를 사용하여 **iOS 기기에서 직접 구동**하고, 그 추론 성능을 검증 및 최적화하기 위해 개발되었습니다.

<br>
<br>

## 🛠️ 1. 개발 환경 설정

성공적인 빌드와 실행을 위해 다음 환경을 준비해야 합니다.

... requirements 확인 중...

<br>
<br>

## 💡 2. 프로젝트 아키텍처 및 데이터 흐름

이 프로젝트는 크게 **Python 기반의 모델 변환 단계**와 **Swift 기반의 모바일 추론 단계**로 나뉩니다.

### 데이터 흐름 요약

1.  **모델 준비 (Python)**: 학습된 EoMT 모델 $\rightarrow$ **`convert_fixed3.py`** $\rightarrow$ **`EoMT_2.mlpackage`** (Core ML 파일)
2.  **모바일 실행 (Swift)**: iOS App $\rightarrow$ **`modelhandler.swift`** (로드) $\rightarrow$ **`EoMTEval.swift`** (추론) $\rightarrow$ 결과 표시

<br>
<br>

## 📂 3. 주요 모듈 설명

프로젝트 파일들은 역할에 따라 Python 스크립트와 Swift 코드로 나뉩니다.

### A. Python (모델 변환 및 검증)

| 파일 | 주요 역할 | 상세 설명 |
| :--- | :--- | :--- |
| **`convert_fixed3.py`** | **Core ML 변환기** | 학습된 **EoMT 모델**의 체크포인트를 불러와 Core ML의 **`.mlpackage`** 포맷으로 최적화하여 변환합니다. |
| **`mlpackage_inference.py`** | **변환 검증 스크립트** | 변환된 `.mlpackage` 파일을 Python 환경에서 불러와 추론을 수행하고 결과를 확인합니다. |

<br>
<br>

### B. Swift (모바일 앱)

| 파일 | 주요 역할 | 상세 설명 |
| :--- | :--- | :--- |
| **`modelhandler.swift`** | **모델 관리 및 로드** | Core ML 모델 파일(`EoMT_2.mlpackage`)의 로드, 초기화 및 메모리 관리를 담당하는 메인 진입점입니다. |
| **`EoMTEval.swift`** | **추론 로직 (핵심)** | **Core ML 모델을 사용**하여 모바일 환경에서 센서 데이터를 입력받아 실제 추론(Inference)을 실행합니다. |
| **`ContentView.swift`** | **UI 및 화면 표시** | 추론 결과를 화면에 시각적으로 표시하거나, 추론 실행을 위한 버튼 등 사용자 인터페이스 로직을 담당합니다. |

<br>
<br>


## 🏃 4. 실행 및 테스트 방법

...

