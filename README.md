# 🥇 ACK2026.09.22_합성 열화 조건에서 YOLO 기반 신호등 탐지의 시간적 강건성 평가
  
---
### Temporal Robustness Evaluation of YOLO-Based Traffic Light Detection under Synthetic Degradations

- 저자: 최승범, 설재훈, 김동현, 김재원, 오준석, 김영균

기존 객체 탐지 평가는 주로 Precision, Recall, mAP와 같은 프레임 단위 지표에 의존한다. 그러나 연속 주행 환경에서는 신호등을 얼마나 이른 시점부터 안정적으로 탐지하는지와 중요한 적색 신호를 얼마나 오래 연속으로 놓치는지도 중요하다. 본 연구에서는 DTLD 주행 시퀀스에서 YOLOv8n, YOLO11n, YOLO12n의 예측을 Track 단위로 연결하고, AUC5와 적색 신호 연속 완전 미검출 지속시간 \(\bar D\)를 이용해 합성 열화 조건에서의 시간적 강건성을 비교하였다.

---

# 1. 데이터 셋

Occupancy Networks에서 제안된 프로토콜을 통해 전처리된 대규모 3D 객체 데이터 셋 ShapeNet Core 활용

  **Train/Validation/Test**

  > RGB Image : 35024/4378/4381 개 (.jpg), 24개의 시점에서 렌더링 된 객체 이미지
  >
  > Ground-Truth : 35024/4378/4381 개 (.npz), 3D 좌표와 물체 내 존재 여부를 나타내는 이진 데이터 쌍
  >
  > Camera parameter : 35024/4378/4381 개 (.npz)
  
# 2. 문제 제기

**필요성**

![Occlusion Surface](https://github.com/seoljaehun/ACK2026.05.22_3D_Shape_Restoration/blob/main/Image_Data/Occlusion%20Surface.PNG)

- 로봇이 물체를 안정적으로 파지하려면 표면 구조와 6D 공간 자세를 반영한 정밀한 3차원 형상 인식이 필수적
- 실제 환경에서는 단일 카메라 시점에 의존하기 때문에, 가려진 뒷면의 기하 정보 손실이 빈번하게 발생

**기존 방식 문제**

![Spectral Bias](https://github.com/seoljaehun/ACK2026.05.22_3D_Shape_Restoration/blob/main/Image_Data/Spectral%20Bias.PNG)

- 기존 MLP 기반 딥러닝 모델들은 저주파 성분을 먼저 학습하는 Spectral Bias 특성을 지님
- 이로 인해 전반적인 형태는 잘 잡지만, 모서리나 굴곡 같은 고주파 세부 기하 구조가 뭉툭하게 표현(Over-smoothing)되는 뚜렷한 한계가 존재

# 3. 알고리즘 구조

단일 시점 이미지로부터 보이지 않는 영역을 포함한 전체 3D 형상을 정밀하게 복원

![Overall System Process](https://github.com/seoljaehun/ACK2026.05.22_3D_Shape_Restoration/blob/main/Image_Data/Overall%20System%20process_2.png)

**Feature Extraction**

  - CNN 기반 인코더를 거쳐 물체의 시각적 윤곽을 파악하고 전역/지역 특징을 추출
  - 이산 웨이블릿 변환(DWT, Level 2)을 통해 이미지의 엣지와 텍스처 등 고주파 성분만을 분해 및 정제
 
**Decoder**

  - 큰 뼈대를 구축하는 Global Decoder와 미세한 디테일을 살리는 Local Decoder가 병렬로 작동
  - 두 디코더의 결과값을 합산하여 3차원 공간의 점유 확률을 예측하고 최종 3D 메쉬 형상을 생성

# 4. 세부 알고리즘 구현

**Spatial & Frequency Feature Extraction**

  - **Spatial Branch** :
![Spatial Feature Extraction](https://github.com/seoljaehun/ACK2026.05.22_3D_Shape_Restoration/blob/main/Image_Data/Spatial%20Feature%20Extraction.png)
    - 잔차 연결을 통해 안정적인 학습이 가능하고 연산 효율이 높은 ResNet-18을 인코더 백본으로 채택
    - 초기 레이어의 세밀한 시각 패턴(지역 특징)부터 심층 레이어의 전체적인 윤곽까지 단계적으로 통합되어 1D 전역 특징 도축
    - 각 계층의 중간 레이어에서 다중 스케일의 2D 지역 특징(Local Features)을 추출
 
  - **Frequency Branch** :
![Frequency Feature Extraction](https://github.com/seoljaehun/ACK2026.05.22_3D_Shape_Restoration/blob/main/Image_Data/Frequency%20Feature%20Extraction.png)
    - 공간-주파수 국소화 특성이 우수한 Level 2 2D-DWT를 도입하여 이미지를 주파수 대역별로 분해
    - 저주파 성분(LL)은 배제하고, 미세 표면 굴곡 복원에 집중하기 위해 RGB 채널별 고주파 성분(LH, HL, HH)만을 선택적으로 결합하여 엣지와 텍스처 정보를 추출 및 정제

**Double Track Decoder**

  - **Global Decoder** :
![Global Decoder](https://github.com/seoljaehun/ACK2026.05.22_3D_Shape_Restoration/blob/main/Image_Data/Global%20Decoder.PNG)
    - 전역 문맥을 반영하여 3D 물체의 전체적인 형태와 기본 뼈대(Baseline)를 구성하는 전역 Logit을 생성
    - 3D 공간 좌표가 다수의 ResNet 블록을 통과할 때 인코더의 1D 전역 특징을 조건으로 주입하는 CBN을 적용하여, 공간 특징을 전체 형상에 맞게 동적으로 변조


  - **Local Decoder** :
![Local Decoder](https://github.com/seoljaehun/ACK2026.05.22_3D_Shape_Restoration/blob/main/Image_Data/Local%20Decoder.PNG)
    - 주파수 특징이 다층 신경망(MLP)과 시그모이드 함수를 거쳐 포인트 단위의 어텐션 가중치를 산출
    - 산출된 가중치를 2D 지역 특징에 요소별로 곱하여($\odot$) 형상 복원에 유의미한 미세 표면 굴곡 및 디테일 영역만을 선택적으로 강화
    - 본 뼈대를 기하학적으로 보완해 줄 고주파수 디테일 잔차로 동작
   
**Implicit Occupancy Representation**

  - 고정 해상도 제약을 극복하기 위해, 연속적인 3D 공간 좌표($p \in R^3$)의 점유 확률을 학습하는 방식
  - 전역 위상을 잡는 Global Decoder와 디테일을 가감하는 Local Decoder의 예측값(Logit)을 합산하여 최종 점유 확률을 도출
 
**3D Mesh Reconstruction**

  - 최종 결합된 점유 확률 공간 볼륨에 결정 경계를 적용하고, Marching Cubes 알고리즘을 적용하여 정교하고 매끄러운 최종 3D 메쉬를 생성

# 5. 실험 결과
제안 모델을 기존 Baseline, 정답 GT와 시각적으로 비교하고, 평가지표를 통해 성능 향상을 정량적으로 입증

![3D Shape Result](https://github.com/seoljaehun/ACK2026.05.22_3D_Shape_Restoration/blob/main/Image_Data/3D%20Shape%20Result_hori.png)

![3D Shape Metrics](https://github.com/seoljaehun/ACK2026.05.22_3D_Shape_Restoration/blob/main/Image_Data/Metrics.PNG)

- 단일 전역 특징 기반의 Baseline 모델이 빈 공간을 과도하게 평활화하여 메워버린 반면, 제안 기법은 복잡한 위상 구조의 미세한 디테일과 빈 공간을 성공적으로 복원
- 표면 간 최소 거리 오차를 나타내는 핵심 지표인 Chamfer Distance(CD)를 0.0192에서 0.0161로 약 16.32% 감소
- Volumetric IoU와 1% 거리 오차 내 Point Matching F-Score를 각각 1.13%p, 4.90%p 향상
- Spectral Bias로 인해 학습하기 어려운 고주파 성분까지 정교하게 복원함으로써, 기존 모델의 한계를 극복하고 정밀 작업 적용에 대한 유효성을 입증

# 6. 결론

![Camera Ray](https://github.com/seoljaehun/ACK2026.05.22_3D_Shape_Restoration/blob/main/Image_Data/Camera%20Ray.png)

- 복잡한 위상 구조의 미세한 디테일을 성공적으로 복원하고 평가지표를 크게 향상시켜 로봇 파지 등 정밀 작업에서의 유효성을 입증
- 단일 시점 정보 의존으로 인한 깊이 모호성(Depth Ambiguity) 때문에 복원된 객체가 카메라 광선 방향을 따라 길게 늘어지는 현상이 일부 관찰
- 향후 이러한 기하학적 왜곡을 해결하기 위해 객체의 특징을 여러 직교 평면에 투영하여 다각도로 통합하는 Multi-Plane 표현 방식 기반의 연구를 진행할 계획

---

# 관련 자료

- Paper : <https://kiss.kstudy.com/Detail/Ar?key=4254299>
- Dataset : <https://shapenet.org>, <https://github.com/autonomousvision/occupancy_networks>
- 참고 문헌 : <https://github.com/seoljaehun/ACK2026.05.22_3D_Shape_Restoration/blob/main/Reference/%EC%B0%B8%EA%B3%A0%EB%AC%B8%ED%97%8C>

---
