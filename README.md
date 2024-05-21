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
우리는 공개된 마더보드 결함 탐지 데이터셋을 사용할 계획이다. 이 데이터셋은 컴퓨터 메인보드 결함을 감지하기 위한 객체 감지 모델에 사용되며, YOLOv5, YOLOv7, YOLOv8 및 CLIP 포맷으로 제공된다. 데이터셋에는 1000개 이상의 항목과 2800개의 주석이 포함되어 있으며, 일반적인 메인보드 결함(예: 느슨한 나사, 잘못 사용된 나사, 분리된 CPU 팬 포트, 긁힘 등)에 대한 정보가 담겨 있다.

### Dataset Details
이미지 수: 389
주석 수: 2860
클래스 수: 11
평균 이미지 크기: 1.10MP
중앙값 이미지 비율: 1044x1074
이 데이터셋은 메인보드 생산 결함을 정확히 감지하고 분석하는 데 유용하며, 인공지능 모델의 학습과 평가에 적합한 환경을 제공한다.



### 데이터셋 통합 및 처리
위의 가져온 데이터셋들은 모두 경기도 내의 읍/면/동에 대한 정보를 가지고 있다 따라서 경기도내의 읍/면/동을 기준으로 그룹핑 하여 데이터셋들을 통합할 것이다. 
이때 알고가야 할 점은 데이터셋마다 읍/면/동에 대한 정보는 가지고 있지만 저장 형식에 차이가 있기 때문에 형식을 통일하고 그룹핑 할 것이다.  

# Datasets
- Describing your dataset

# Methodology
- Explaining your choice of algorithms (methods)
- Explaining features (if any)

# Evaluation & Analysis
- Graphs, tables, any statistics (if any)

# Related Work (e.g., existing studies)
- Tools, libraries, blogs, or any documentation that you have used to do this project.

# Conclusion: Discussion
