# AI-X-DeepLearning (YOLOv7_motherboard)
> 동영상 링크: 
## 목차

1. [Proposal](#i-proposal)
2. [Datasets](#ii-datasets)
3. [Methodology](#iii-methodology)
4. [Evaluation & Analysis](#iv-evaluation--analysis)
5. [Related Work](#v-related-work)
6. [Conclusion: Discussion](#vi-conclusion-discussion)
   
## Title: YOLOv7를 활용한 컴퓨터 메인보드 불량 검증
## Members
- 남현기 | 한양대 에리카 컴퓨터학부 nam5286@hanyang.ac.kr
- 홍진수 | 한양대 에리카 컴퓨터학부 h1101201@hanyang.ac.kr


## I. Proposal
### Motivation
누구나 전자 제품을 사용하다가 결함이 있는 제품을 접한 경험이 있다. 이런 결함은 생산 과정에서 발생하는 경우가 많으며, 이를 미리 감지하고 수정하는 것이 중요하다. 특히, 메인보드와 같은 전자 제품의 경우 결함이 발생하면 아예 작동이 안되는 경우가 있다.<br>
메인보드 결함 탐지 데이터셋은 비교적 특징들이 뚜렷하고 많기 때문에 이러한 결함을 자동으로 감지하고 분석하는 모델을 만드는데 적합할 것으로 예상되어 이를 주제로 정했다.

### Goal
주어진 메인보드 이미지에서 YOLOv7을 활용하여 결함을 자동으로 탐지하고 분류하는 모델을 만드는 것이 목표다. 이를 통해 생산 과정에서 발생할 수 있는 결함을 미리 발견하고 수정함으로써 제품의 품질을 높일 수 있을 것으로 기대한다. 

```
데이터셋 링크  
컴퓨터 메인보드 결함: 결함이 있는 메인보드의 이미지와 레이블을 담고 있다
- https://www.kaggle.com/datasets/yuelinxin/computer-motherboard-production-defects

```

## II. Datasets
위 링크에 명시된 마더보드 결함 탐지 데이터셋을 사용할 계획이다. 이 데이터셋은 컴퓨터 메인보드 결함을 감지하기 위한 객체 감지 모델에 사용되며, YOLOv5, YOLOv7, YOLOv8 및 CLIP 포맷으로 제공된다. 데이터셋에는 1000개 이상의 이미지와 2800개의 주석이 포함되어 있으며, 일반적인 메인보드 결함(예: 느슨한 나사, 잘못 사용된 나사, 분리된 CPU 팬 포트, 긁힘 등)에 대한 정보가 담겨 있다.

### Dataset Details

![--2022-03-08-21-44-03_png rf 90d11f761f078ae435fa32ae00b6750d](https://github.com/namnhong/YOLOv7_motherboard/assets/50573818/c22de529-6f1f-49db-8549-67bfda57538c)

이미지 수: 389<br>
주석 수: 2860<br>
클래스 수: 11<br>
평균 이미지 크기: 1.10MP<br>
중앙값 이미지 비율: 1044x1074<br>
<br>
![image](https://github.com/namnhong/YOLOv7_motherboard/assets/50573818/d216ad14-461f-40d9-821d-bfdd5cc770fd)

### Dataset 전처리
각 이미지에 다음과 같은 전처리가 적용되었다.<br>
<br>
픽셀 데이터의 자동 방향 조정 (EXIF 방향 정보 제거)<br>
1000x1000으로 크기 조정 (Stretch)<br>
자동 대비 조정 (adaptive equalization 사용)<br>
<br>
### Dataset 증강
각 원본 이미지에서 아래와 같은 3가지 증강 기법이 적용되었다.<br>
<br>
50% 확률로 수평 반전<br>
하나를 동일한 확률로 적용 (없음, 시계 방향 90도 회전, 반시계 방향 90도 회전)<br>
랜덤 노출 조정: -25%에서 +25% 사이<br>
<br>
### Dataset 분배
훈련(train) 데이터: 939장 (92.51%)<br>
검증(valid) 데이터: 31장 (3.05%)<br>
테스트(test) 데이터: 45장 (4.43%)<br>

<br>

### Dataset Label Details
데이터셋의 레이블은 다음과 같이 구성되어 있다.<br>

CPU_FAN_NO_Screws: CPU 팬에 나사가 없음<br>
CPU_FAN_Screw_loose: CPU 팬 나사가 느슨함<br>
CPU_FAN_Screws: CPU 팬 나사가 정상적으로 부착됨<br>
CPU_fan: CPU 팬<br>
CPU_fan_port: CPU 팬 포트<br>
CPU_fan_port_detached: CPU 팬 포트가 분리됨<br>
Incorrect_Screws: 잘못된 나사 사용<br>
Loose_Screws: 느슨한 나사<br>
No_Screws: 나사가 없음<br>
Scratch: 긁힘<br>
Screws: 나사<br>

## III. Methodology

### YOLO
YOLO는 You Only Look Once의 약자로 객체 검출(Object detection) 알고리즘의 한 종류로, 이미지나 비디오 프레임에서 여러 객체를 실시간으로 검출할 수 있다. YOLO는 이미지 전체를 한 번에 처리하여 객체를 예측하는 방식을 사용하므로, 다른 객체 검출 알고리즘에 비해 매우 빠르고 효율적이다.<br>

<br>

### YOLOv7
YOLOv7은 YOLO 시리즈의 최신 버전으로, 여러 가지 기술적 개선을 통해 더 높은 정확도와 효율성을 제공한다.<br>

<br>

### YOLOv7 architecture
- 전체 구조<br>
![The-structure-of-YOLOv7](https://github.com/namnhong/YOLOv7_motherboard/assets/55042341/06d2390e-484d-40f1-a82f-631d4491d214)<br>
<br>

- E-ELAN(Extended efficient Layer Aggregation Networks) 확장<br>
![image](https://github.com/namnhong/YOLOv7_motherboard/assets/55042341/f4ead03e-a86d-4a6a-950a-4edf33d99349)<br>
그레디언트가 경로를 효율적으로 제어(확장, 셔플, 병합)하는 E-ELAN 구조를 제안했다.<br>
<br>

- 복합 스케일링<br>
![image](https://github.com/namnhong/YOLOv7_motherboard/assets/55042341/89376215-4695-4dac-86a5-9f0f3b0b772b)<br>
기존의 깊이 스케일링이 수행될 때는 블록의 출력 폭도 증가했다. 이는 뒤쪽의 입력 폭도 같이 증가하는 결과가 발생했다.<br><br>
![image](https://github.com/namnhong/YOLOv7_motherboard/assets/55042341/07f208eb-aeb6-46b2-be70-f7fe2bd31442)<br>
위와 같이 입/출력 폭을 유지하면서 깊이만 스케일하도록 하는 모델을 제안해 계산량을 감소시켰다.<br>
<br>

- Planned Re-parameterized convolution<br>
매개변수가 변경된 layer가 있을 때, residual 연결이 있으면 안된다는 것을 발견해 Planned Re-parameterized convolution을 제안했다.
inference 비용을 늘리지 않고 정확도를 향상시키는 방법이다.<br>
<br>

- Coarse for auxiliary and fine for lead loss<br>
![image](https://github.com/namnhong/YOLOv7_motherboard/assets/55042341/aaba55af-47c9-4de5-8eb8-ccbe79a1a386)<br>
Lead Head가 학습에 더 많은 영향을 미치므로 Aux Head라 할지라도 Lead Head를 거친 것을 사용하는 것이 성능에 더 좋다는 주장으로 위와 같은 방법을 제안했다.<br>

<br>

### YOLOv7 performance
- 실시간 객체 검출 모델 간 성능 평가<br>
![image](https://github.com/namnhong/YOLOv7_motherboard/assets/55042341/890163a3-7f04-4409-877c-f7ed11348172)<br>
YOLOv7 모델의 평가는 비교 가능한 다른 실시간 객체 검출 모델들보다 더 빠르고(x축) 더 높은 정확도(y축)로 추론한다는 것을 보여준다.<br>
<br>

- YOLOv7 모델의 버전 간 성능 평가<br>
![image](https://github.com/namnhong/YOLOv7_motherboard/assets/50573818/7bd3a0b2-c295-41f4-93f4-5283de10d000)
<br>
YOLOv7은 annotation quality가 낮아져도 YOLOv4보다 성능(AP 및 mAP)이 더 높게 유지되며, 더 견고한 성능을 보인다.<br>
<br>

## IV. Evaluation & Analysis

### 학습 과정

100 epoch 동안 YOLOv7 모델을 학습하면서 기록을 아래에 남겼다.<br>
각 에포크에서의 손실 값과 주요 지표들

| Epoch | GPU 사용량 | 총 손실 | Box 손실 | cls 손실 | obj 손실 | mAP@0.5 | 정밀도 | 재현율 | val Box | val Obj | val cls |
|-------|------------|--------|---------|----------|---------|---------|--------|--------|--------|--------|--------|
| 0/99  | 1.61G      | 0.06916 | 0.03141 | 0.03306  | 0.1336  | 30      | 1024   | 0.6822 | 0.1238 | 0.1361 | 0.06081 |
| 1/99  | 16.6G      | 0.05363 | 0.02406 | 0.02238  | 0.1001  | 17      | 1024   | 0.8607 | 0.1418 | 0.1493 | 0.07374 |
| ...   | ...        | ...     | ...     | ...      | ...     | ...     | ...    | ...    | ...    | ...    | ...    |
| 99/99 | 16.6G      | 0.01403 | 0.007901| 0.0002935| 0.02222 | 23      | 1024   | 0.8433 | 0.9331 | 0.8457 | 0.5683  |

학습 과정에서 손실 값은 지속적으로 감소하며, 모델의 정밀도와 재현율은 증가했다. 최종 에포크에서는 총 손실이 0.01403으로 매우 낮고, mAP@0.5는 0.9331로 높은 성능을 보여준다.

### 하이퍼파라미터 설정

모델 학습에 사용된 주요 하이퍼파라미터는 다음과 같다.

- 초기 학습률 (lr0): 0.01
- 최종 학습률 (lrf): 0.1
- 모멘텀 (momentum): 0.937
- 가중치 감소 (weight_decay): 0.0005

  
### 모델 학습 결과 및 평가

#### Confusion Matrix<br>
![confusion_matrix](https://github.com/namnhong/YOLOv7_motherboard/assets/50573818/5d5eeecb-9b55-4bc9-9384-e9e67685e5dd)


Confusion Matrix을 통해 모델의 클래스별 정확도를 분석한 결과, 대부분의 클래스에서 높은 정확도와 재현율을 보였다.<br>
특히 대부분의 클래스에서 매우 높은 정확도를 나타나는 것을 확인 가능하다.<br>
그러나 `CPU_fan_port_detached` 클래스는 비교적 낮은 정확도를 보였으며, 이는 추가적인 데이터로 학습이 더욱 더 필요할 것 같다.

#### Precision-Recall Curve
![PR_curve](https://github.com/namnhong/YOLOv7_motherboard/assets/50573818/773f0e2f-998f-4a76-81ef-d48f69b13c1e)
Precision-Recall(PR) Curve를 통해 모델의 전반적인 성능을 평가할 수 있다.<br>
대부분의 클래스에서 높은 정밀도와 재현율을 유지하며, 안정적인 성능이 나온다는 것을 알 수 있다.

#### F1 Score
![F1_curve](https://github.com/namnhong/YOLOv7_motherboard/assets/50573818/483c28ea-e2b2-471d-9da6-2126a64ff00a)

F1 Score는 정밀도와 재현율의 조화평균을 기준으로 모델의 균형 잡힌 성능을 평가한다.<br>
대부분의 클래스에서 F1 점수가 높게 나타났으며, 역시 괜찮은 성능을 보임을 나타낸다.

### 테스트 결과 및 이미지 분석 결과
테스트 데이터셋에서의 모델 성능은 다음과 같다

- 평균 정밀도(mAP@0.5): 0.933
- 정확도 (Precision): 0.8433
- 재현율 (Recall): 0.9331
- F1 Score: 0.87

아래 이미지는 YOLOv7 모델을 학습하는 동안 메인보드 결함을 탐지한 결과이다. 각 이미지에는 예측된 결함이 박스와 라벨로 표시되어 있다. 

#### 첫 번째 배치

| 라벨 이미지 | 예측 이미지 |
|-------------|-------------|
| ![test_batch0_labels](https://github.com/namnhong/YOLOv7_motherboard/assets/50573818/d4fb154e-627d-4031-9e06-149c0cb62b36)| ![test_batch0_pred](https://github.com/namnhong/YOLOv7_motherboard/assets/50573818/97f7da1c-2681-45ad-aedf-3479c5306ee1) |

#### 두 번째 배치

| 라벨 이미지 | 예측 이미지 |
|-------------|-------------|
| ![test_batch1_labels](https://github.com/namnhong/YOLOv7_motherboard/assets/50573818/74105ac3-d25f-4cc6-9496-cf6a143dee22)| ![test_batch1_pred](https://github.com/namnhong/YOLOv7_motherboard/assets/50573818/5f2574dc-c7b0-4088-bcd4-c49793ed5f33)|

#### 결과 해석
모델은 다양한 결함을 정확하게 탐지하고 라벨링하는 데 성공하였다.<br>
대부분의 클래스에서 예측이 정확하게 이루어졌다. 다만 몇몇 이미지는 예측에서 약간의 오차가 있었지만, 전반적인 성능은 우수하다. 
<br>


## V. Related Work (e.g., existing studies)

```
1. 데이터셋 링크  
컴퓨터 메인보드 결함: 결함이 있는 메인보드의 이미지와 레이블을 담고 있다
- https://www.kaggle.com/datasets/yuelinxin/computer-motherboard-production-defects
2. 모델 학습 및 성능 테스트 참고 게시물
YOLOv7 커스텀 데이터 전반적인 학습 방법
- https://tae-jun.tistory.com/11
3. 그래프 해석
분류 성능 지표 Precision(정밀도), Recall(재현율), F1-score 설명
- https://ai-com.tistory.com/entry/ML-%EB%B6%84%EB%A5%98-%EC%84%B1%EB%8A%A5-%EC%A7%80%ED%91%9C-Precision%EC%A0%95%EB%B0%80%EB%8F%84-Recall%EC%9E%AC%ED%98%84%EC%9C%A8

```


## VI. Conclusion: Discussion

### 결과 요약
1. 학습 과정에서 mAP@0.5가 **0.9331**을 기록하며, 모델의 전반적인 성능이 매우 우수함을 보여주었다. 특히, `CPU_fan_port_detached`를 제외한 모든 클래스에서 높은 정확도를 보였다.
2. 초기 학습률, 모멘텀, 가중치 감소 등의 최적의 하이퍼파라미터 설정을 통해 모델의 학습 안정성과 성능 향상을 이끌었다.
3. 메인보드 데이터셋에 다양한 증강 기법과 전처리 기법을 적용하여 모델의 일반화 성능을 향상시켰다.

### 한계 및 향후 연구 방향
1. 현재 사용된 데이터셋은 메인보드의 특정 결함에만 집중되어 있다. 더 다양한 결함을 포함하도록 레이블을 데이터셋에 추가하여 모델의 범용성을 높일 수 있을 것이다.
2. `CPU_fan_port_detached` 클래스에서 비교적 낮은 정확도를 보인 점을 개선하기 위해, 해당 클래스에 대한 추가적인 데이터 증강과 데이터셋이 필요하다.


이번 프로젝트에서 YOLOv7 모델을 사용하여 컴퓨터 메인보드의 결함을 자동으로 탐지하고 분류하는 우수한 성능의 모델을 만들게 되었다.<br>
추가적인 데이터 증강과 모델 튜닝을 통해 지금보다 더 나은 성능을 기대할 수 있을 뿐만 아니라, 이를 실제 생산 환경에서도 활용할 수 있을 것이라고 예상이 된다.
