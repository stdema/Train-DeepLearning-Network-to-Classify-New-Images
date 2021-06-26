# Train-DeepLearning-Network-to-Classify-New-Images
 - 새로운 영상 세트를 분류하기 위해 googlenet의 마지막 신경망 층을 새롭게 바꾼 뒤 훈련시켰다.

이 예제에서는 새로운 영상 세트를 분류할 수 있도록 전이 학습을 사용하여 컨벌루션 신경망을 다시 훈련시키는 방법을 보여준다.
사전 훈련된 신경망은 1백만 개가 넘는 영상에 대해 훈련되었으며 1000가지 사물 범주로 분류할 수 있다. 분류 신경망은 다양한 영상을 대표하는 다양한 특징을 학습했다.
일반적으로 전이 학습으로 신경망을 미세 조정하는 것이 무작위로 초기화된 가중치를 사용하여 신경망을 처음부터 훈련시키는 것보다 훨씬 쉽고 빠르다.

### 데이터 불러오기
```c
unzip('MerchData.zip');
imds = imageDatastore('MerchData', ...
    'IncludeSubfolders',true, ... //하위 폴더의 파일들까지 전체 저장
    'LabelSource','foldernames'); //라벨명을 폴더명으로 입력
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7); //train/validation 데이터 0.7 비율로 나눔(저장형식은 imagedatastore 그대로)
```
![image](https://user-images.githubusercontent.com/86040099/123505580-ea59e680-d69a-11eb-89e4-0c8351aa8797.png)

[데이터 셋. 한 폴더 당 15개의 이미지가 있다. label : 'Cap', 'Cube', 'Playing Cards', 'Screwdriver', 'Torch']

### 사전 훈련된 신경망 불러오기
```c
net = googlenet;
analyzeNetwork(net) //딥러닝 신경망 분석기
```
![image](https://user-images.githubusercontent.com/86040099/123505139-48d19580-d698-11eb-96a0-2034ee7d14d3.png)

[딥러닝 신경망 분석기에서 googlenet 구조]

```c
net.Layers(1) //첫번째 층 출력 – 입력 층. inputSize[224 224 3] 3은 색 채널의 개수
inputSize = net.Layers(1).InputSize;
```
![image](https://user-images.githubusercontent.com/86040099/123505570-d9a97080-d69a-11eb-90ff-90887c9c5255.png)
                      
### 마지막 계층 바꾸기
신경망의 컨벌루션 계층은 마지막 학습 가능한 계층과 마지막 분류 계층이 입력 영상을 분류하는 데 사용하는 영상 특징을 추출한다. GoogLeNet의 두 계층 'loss3-classifier'와 'output'은 신경망이 추출하는 특징을 클래스 확률, 손실 값 및 예측된 레이블로 조합하는 방법에 대한 정보를 포함한다.
```c
if isa(net,'SeriesNetwork')         //isa : 입력값이 지정된 데이터형을 갖는지 확인/ SeriesNetwork : 계층이 하나씩 차례대로 연결. 하나의 입력계층과 하나의 출력 계층
  lgraph = layerGraph(net.Layers);  //신경망에서 계층 그래프 추출
else
  lgraph = layerGraph(net);
end 
```
![image](https://user-images.githubusercontent.com/86040099/123505625-2db45500-d69b-11eb-8b1a-c463a45840d6.png)

```c
[learnableLayer,classLayer] = findLayersToReplace(lgraph);  //바꿀만한 층을 찾는 함수.
[learnableLayer,classLayer]                                
```
![image](https://user-images.githubusercontent.com/86040099/123505816-25a8e500-d69c-11eb-9894-0193eb8c25a7.png)

[learnableLayer : loss3-classifier, classLayer : output]

대부분의 신경망에서, 학습 가능한 가중치를 갖는 마지막 계층은 완전 연결 계층이다. 이 완전 연결 계층을 출력값의 개수가 새 데이터 세트의 클래스 개수와 같은(이 예제에서는 5) 새로운 완전 연결 계층으로 바꾼다. 

```c
numClasses = numel(categories(imdsTrain.Labels));   //numel : 배열 요소의 전체 수/ categories : 카테고리형 배열 만들기(겹치는 것 없앰) => train 라벨의 수 지정

%새로운 완전연결계층 만들기 fullyConnectedLayer(출력 크기, 파라미터):입력값*가중치 행렬+편향 벡터
if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')   //fullyconnectedlayer 완전 연결 계층
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...            //이름 지정
        'WeightLearnRateFactor',10, ... //가중치 초기화
        'BiasLearnRateFactor',10);      //편향 초기화
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer') //2차원 컨벌루션 계층
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
End

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);  //lgraph 기존의 완전 연결 계층을 새롭게 만든 계층으로 바꿈

newClassLayer = classificationLayer('Name','new_classoutput'); //새로운 output계층 만듦
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);   //lgraph 기존의 출력 계층을 새롭게 만든 출력 계층으로 바꿈

figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);     //그래프 위치 조정
plot(lgraph)
ylim([0,10])
```
![image](https://user-images.githubusercontent.com/86040099/123506084-5dfcf300-d69d-11eb-965d-9bfe18dfeb16.png)

['loss3-classifier' 와 'output' 두 층이 'new_fc', 'new_classoutput'으로 바뀐 그래프]

### 초기 계층 고정하기
새로운 영상 세트를 사용하여 신경망을 다시 훈련시킬 준비가 되었다. 선택적으로 신경망의 앞쪽 계층의 학습률을 설정하여 이전 계층의 가중치를 고정할 수 있다. 훈련 중에 trainNetwork는 고정된 계층의 파라미터를 업데이트하지 않는다. 고정된 계층의 기울기를 계산할 필요가 없으므로 전반부에 있는 여러 계층의 가중치를 고정하면 신경망 훈련 속도를 대폭 높일 수 있다.

```c
layers = lgraph.Layers;           //lagraph의 레이어로 새로 연결
connections = lgraph.Connections; 

layers(1:10) = freezeWeights(layers(1:10));                //처음 10개 계층 학습률 0으로 설정
lgraph = createLgraphUsingConnections(layers,connections); //원래 순서대로 다시 연결
```

### 신경망 훈련시키기
이 신경망의 입력 영상은 크기가 224x224x3이어야 하는데 영상 데이터저장소의 영상은 이와 크기가 다르다. 증대 영상 데이터저장소를 사용하여 훈련 영상의 크기를 자동으로 조정한다. 훈련 영상에 대해 추가로 수행할 증대 연산을 지정한다. 즉, 세로 축을 따라 훈련 영상을 무작위로 뒤집고, 최대 30개의 픽셀을 최대 10%까지 가로와 세로 방향으로 무작위로 평행 이동한다. 데이터 증대는 신경망이 과적합되는 것을 방지하고 훈련 영상의 정확한 세부 정보가 기억되지 않도록 하는 데 도움이 된다.
```c
pixelRange = [-30 30];  //과적합 방지
scaleRange = [0.9 1.1]; //영상을 무작위로 뒤집고 무작위로 평행이동
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
    
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);  //train과 val영상 크기 조정
```

훈련 옵션을 지정한다. 아직 고정되지 않은 전이된 계층의 학습을 늦추기 위해 InitialLearnRate를 작은 값으로 설정한다. 
훈련을 진행할 Epoch의 횟수를 지정한다. 전이 학습을 수행할 때는 많은 횟수의 Epoch에 대해 훈련을 진행하지 않아도 된다. Epoch 1회는 전체 훈련 데이터 세트에 대한 하나의 완전한 훈련 주기를 의미한다. 미니 배치 크기와 검증 데이터를 지정합니다. Epoch당 한 번씩 검증 정확도를 계산한다.
```c
miniBatchSize = 10;

valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize); //floor : 내림
options = trainingOptions('sgdm', ...                          //sgdm : 모멘텀(한점의 기울기)을 사용한 확률적 경사하강법 
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');
[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)

net = trainNetwork(augimdsTrain,lgraph,options);   //훈련 옵션을 토대로 훈련시킴

```
![image](https://user-images.githubusercontent.com/86040099/123506413-02cc0000-d69f-11eb-8847-774d065af779.png)

[데이터가 적기 때문에 훈련이 빠르게 진행됨.]

 ### 검증 영상 분류하기
미세 조정된 신경망을 사용하여 검증 영상을 분류한 다음 분류 정확도를 계산합니다.
```c
[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)
```
![image](https://user-images.githubusercontent.com/86040099/123506440-34dd6200-d69f-11eb-9f28-794c144223d1.png)

[정확도 95%]

```c
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end
```

![image](https://user-images.githubusercontent.com/86040099/123506513-8259cf00-d69f-11eb-9af2-077b59e1d1f6.png)

[4개의 샘플 검증 영상을 예측된 레이블 및 이 레이블을 갖는 영상의 예측된 확률과 함께 표시함]
