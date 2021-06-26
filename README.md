# Train-DeepLearning-Network-to-Classify-New-Images
 - 새로운 영상 세트를 분류하기 위해 googlenet의 마지막 신경망 층을 새롭게 바꾼 뒤 훈련시켰다.

이 예제에서는 새로운 영상 세트를 분류할 수 있도록 전이 학습을 사용하여 컨벌루션 신경망을 다시 훈련시키는 방법을 보여준다.
사전 훈련된 신경망은 1백만 개가 넘는 영상에 대해 훈련되었으며 1000가지 사물 범주로 분류할 수 있다. 분류 신경망은 다양한 영상을 대표하는 다양한 특징을 학습했다.
일반적으로 전이 학습으로 신경망을 미세 조정하는 것이 무작위로 초기화된 가중치를 사용하여 신경망을 처음부터 훈련시키는 것보다 훨씬 쉽고 빠르다.

### 데이터 불러오기
```
unzip('MerchData.zip');
imds = imageDatastore('MerchData', ...
    'IncludeSubfolders',true, ... %하위 폴더의 파일들까지 전체 저장
    'LabelSource','foldernames'); %라벨명을 폴더명으로 입력
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7); %train/validation 데이터 0.7 비율로 나눔(저장형식은 imagedatastore 그대로)
```

### 사전 훈련된 신경망 불러오기
