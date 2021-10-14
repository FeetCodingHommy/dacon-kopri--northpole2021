# \[데이콘\] 북극 해빙예측 AI 경진대회

![HOME](https://user-images.githubusercontent.com/38153357/123811629-61f37400-d92e-11eb-880d-bc54c3ef1e7f.png)

* **ELMo 구조를 차용한 양방향성 ConvLSTM Seq2Seq**



## 대회 공지사항

### 주제 및 배경 + 대회 설명

- 과거 관측된 해빙 데이터를 통해 앞으로 변화할 해빙을 예측해주세요.
- 1978년부터 관측된 북극 해빙의 변화를 이용하여 **주별** 해빙 면적 변화 예측

### 기한

- 2021.05.10 ~ 2021.06.30 17:59

### 베이스라인 및 [이전 연계 공모전](https://dacon.io/competitions/official/235706/overview/description) 우수작

* 베이스라인 (ConvLSTM 네트워크) : [링크](https://dacon.io/competitions/official/235731/codeshare/2631?page=1&dtype=recent)
* [이전 대회 3위](https://dacon.io/more/interview/112/) (U-Net+ResNet (이미지 분석 네트워크)) : [링크](https://dacon.io/competitions/official/235706/codeshare/2531?page=1&dtype=recent)
* [이전 대회 2위](https://dacon.io/more/interview/113/) (MA 5year (시계열 회귀분석)) : [링크](https://dacon.io/competitions/official/235706/codeshare/2537?page=1&dtype=recent)
* [이전 대회 1위](https://dacon.io/more/interview/114/) (K Nearest Neighbor Regression (비모델 기반, 데이터 기반 회귀)) : [링크](https://dacon.io/competitions/official/235706/codeshare/2523?page=1&dtype=recent)
* [이전 대회 4위부터 15위까지](https://dacon.io/competitions/official/235706/codeshare/?page=1&dtype=recent&ptype=pub)는 ConvLSTM의 변형을 사용



## 내 솔루션

### 경진대회 결과 요약

![4등따리](https://user-images.githubusercontent.com/38153357/137288632-16e05643-02ae-45d1-b92e-443595a5738f.png)

* 점수1(MAE/F1): 5.07로 총 413팀 중 private 4위
* 점수3(F1 score): 0.76으로 Top 5 중 1위

### Baseline 개선 중점

* **학습 속도**
  * 경험적으로 학습 속도가 상당히 느렸음
* **후반 예측값들의 정확도 문제**
  * ConvLSTM seq2seq 모델이 출력하는 연속된 12주 분량의 예측값들 중 **전반부는 정확도가 높을 것으로 추측(가정)**
  * Encoder로부터 Decoder로 전달되는 *하나의 context vector로 부터 12주 분량의 해빙 농도 값을 모두 예측해야 하기 때문*
  * ![baseline_diagram](https://user-images.githubusercontent.com/38153357/137285740-c1daa060-85cc-4834-a09f-3aef9a3563f5.png)
  * context vector로 부터 거리가 먼, 즉 **12주 분량의 예측값들 중 후반부의 정확성을 올릴 수 있는 방법**을 고민

### Baseline 개선 방안 (내 모델)

* 기존 **ConvLSTM 셀**로 이루어진 **Encoder-Decoder 구조**의 시계열 예측 모델을 ***2개*** 생성
* 하나의 모델은 **정방향**으로 입출력이 이루어지며, 다른 하나의 모델은 **역방향**으로 입출력이 이루어짐
* 사전학습 된 **Forward Model**과 **Backward Model**을 합쳐서 **최적의 Weighted Average**를 내놓는 양방향 ConvLSTM Seq2Seq 이미지 모델 구축
  * ![mymodel_diagram](https://user-images.githubusercontent.com/38153357/137285773-ebcaf82e-0072-405a-8f45-4962d44b46e1.png)

### 개선 방안 상세 설명 (변경 및 유지사항)

#### 전처리

* (유지) Baseline 코드는 데이터 중 **최근 약 30년(52주 * 30년) 주별 데이터**의 **첫번째 채널 (해빙 농도)만**을 사용
  * train data: 북극 해빙 변화 데이터 중 최근 1,560개 주별 데이터
  * validation data: train data 중 최근 52개 주별 데이터
* (유지) 과거 12주의 해빙 변화를 보고 2주 뒤부터 12주간 의 변화를 예측하도록 데이터셋 준비

#### 학습

* 세부 파라미터
  * **(변경)** batch_size = 2 → **1**
    * (코드 상 한계로 인해 batch_size 1 이상 계산 불가... ㅠ)
  * **(변경)** hidden_dim = 16 → **24**
  * **(변경)** 활성함수 - Conv2D: 기본(linear) → **tanh**
    * 경험적으로 모델 학습 속도 소폭 증진 확인
* **(변경)** Adam 옵티마이져 (learning_rate = 0.0005) → Adam 옵티마이져 (**유동적인 learning_rate**)
  * epoch 40까지 0.002 부터 0.0005까지 선형적으로 감소, epoch 80까지 0.0005 유지 후 다시 선형적으로 감소
* (유지) MAE 손실 함수 사용

### 학습 결과

#### 학습 과정

* 모델 파라미터: 약 300만
* 소요 시간: 약 2.5일
* 학습 환경: Python 3.7 / TensorFlow 2.3 / Nvidia GeForce RTX 2080 TI

#### Baseline 결과와 개선 모델 결과 비교

* MAE/F1: 약 7.18 → 약 4.76 (public 기준) / 약 5.07 (private 기준)
* MAE: 약 5.25 → 약 3.77 (public 기준) / 약 3.83 (private 기준)
* F1 score: 약 0.73 → 약 0.79 (public 기준) / 약 0.76 (private 기준)

### 의의

* ELMo와 유사한 아키텍쳐를 이미지에 적용하여 모델 성능을 증진 시킨 사례
  * 언어모델 ELMo와 유사한 형태의 Forward Model, Backward Model을 추가하여 모델에 양방향성을 추가하여 예측값 sequence의 양쪽 단 모두 정확도를 높임
  * MAE 및 F1 score 개선에 큰 영향
* 각 ConvLSTM 셀의 Convolution 연산의 activation function을 tanh로 바꾸었을 때의 학습 속도 개선 및 MAE의 더 빠른 감소를 실험적으로 증명
  * epoch 당 소요 시간 감소로 인해 학습 epoch을 늘릴 수 있는 환경 마련
  * 이외에도 selu, elu, leaky_relu 등의 activation function과도 실험적 비교 진행하였음
  * 일종의 scaling 혹은 normalization 효과를 가져온 것으로 추측
* 단일 learning rate에서 epoch에 따라 선형적으로 감소하는 learning rate로 변경
  * 특정 epoch부터 학습이 진행되지 않는 현상 방지. 지속적으로 학습이 원활히 진행

### 한계

* 대회 측 제공한 Baseline ConvLSTM Encoder-Decoder 구조에서 근본적으로 벗어나지는 못함
  * Convolutional Seq2Seq 모델의 Seq2Seq 구조를 Attention 매커니즘으로 대체하여 Seq2Seq 모델 구조적 한계를 벗어나는 방안이 추가로 고려 가능
  * Batch/Layer Normalization 적용 고려 가능
* 각 데이터 포인트의 2~5번째 채널 (위성 관측 불가 영역, 해안선 마스크, 육지 마스크, 결측값)을 활용하지 못함
  * 전처리를 통해 더 우수한 결과 나왔을 수도
* 동 순위 대와 비교해보았을 때 F1 Score는 우수한 편이었으나 MAE가 높은 편