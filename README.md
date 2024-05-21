# AI-X-DeepLearning (YOLOv7_motherboard)
> 동영상 링크: https://youtu.be/TZccazEeNqw
## 목차

1. [Proposal](#i-proposal)
2. [Datasets](#ii-datasets)
3. [Methodology](#iii-methodology)
4. [Evaluation & Analysis](#iv-evaluation--analysis)
5. [Related Work](#v-related-work)
   
## Title: YOLOv7를 활용한 컴퓨터 메인보드 불량 검증
## Members
- 남현기 | 한양대 에리카 컴퓨터학부 nam5286@hanyang.ac.kr
- 홍진수 | 한양대 에리카 컴퓨터학부 h1101201@hanyang.ac.kr


## I. Proposal
### Motivation
누구나 전자 제품을 사용하다가 결함이 있는 제품을 접한 경험이 있다. 이런 결함은 생산 과정에서 발생하는 경우가 많으며, 이를 미리 감지하고 수정하는 것이 중요하다. 특히, 메인보드와 같은 전자 제품의 경우 결함이 발생하면 아예 작동이 안되는 경우가 있다. 메인보드 결함 탐지 데이터셋은 비교적 특징들이 뚜렷하고 많기 때문에 이러한 결함을 자동으로 감지하고 분석하는 모델을 만드는데 적합할 것으로 예상되어 이를 주제로 정했다.

### Goal
우리의 목표는 주어진 메인보드 이미지에서 YOLOv7을 활용하여 결함을 자동으로 탐지하고 분류하는 모델을 만드는 것이다. 이를 통해 생산 과정에서 발생할 수 있는 결함을 미리 발견하고 수정함으로써 제품의 품질을 높일 수 있을 것으로 기대한다. 

```
데이터셋 링크  
컴퓨터 메인보드 결함: 결함이 있는 메인보드의 이미지와 레이블을 담고 있다
- https://www.kaggle.com/datasets/yuelinxin/computer-motherboard-production-defects

```

## II. Datasets
우리는 공개된 마더보드 결함 탐지 데이터셋을 사용할 계획이다. 이 데이터셋은 컴퓨터 메인보드 결함을 감지하기 위한 객체 감지 모델에 사용되며, YOLOv5, YOLOv7, YOLOv8 및 CLIP 포맷으로 제공된다. 데이터셋에는 1000개 이상의 이미지와 2800개의 주석이 포함되어 있으며, 일반적인 메인보드 결함(예: 느슨한 나사, 잘못 사용된 나사, 분리된 CPU 팬 포트, 긁힘 등)에 대한 정보가 담겨 있다.

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
![image](https://github.com/namnhong/YOLOv7_motherboard/assets/55042341/890163a3-7f04-4409-877c-f7ed11348172)<br>
![image](https://github.com/namnhong/YOLOv7_motherboard/assets/55042341/08bfbf54-c157-4224-8b5f-9b9e4909c99e)<br>

## IV. Evaluation & Analysis
- Graphs, tables, any statistics (if any)

## V. Related Work (e.g., existing studies)
- Tools, libraries, blogs, or any documentation that you have used to do this project.

## VI. Conclusion: Discussion
