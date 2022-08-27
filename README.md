# rec_practice
Session-Based Recommendations with Recurrent Neural Network (ICLR 2016)
논문 구현 및 MovieLens 데이터셋을 이용한 코드입니다.

<br/>

## Code
Entry file
- run.py

<br/>

## Data 전처리
MovieLens 데이터셋을 위의 논문에 맞게 전처리하기 위해서는 ratings 데이터를 session 개념으로 나타내어야 합니다. <br/>
단일 영화 평점만으로는 RNN 학습을 할 수 없기 때문에 연속된 영화 평점 기록을 하나의 session으로 만들어 학습을 진행했습니다. <br/>
논문에서 말하는 session은 웹사이트에서 클릭하는 경우을 session으로써 정의하였습니다. <br/>
하지만 영화 평점의 경우 1점부터 5점까지 점수가 있는데, 1점인 경우 영화를 추천받았음에도 불구하고 해당 유저와 잘 안맞은 경우이므로 positive sample이라 할 수 있을지 의문이 들었습니다. <br/>
따라서 평점과 상관없이 일단 영화를 본 것 자체를 해당 영화에 관심있었다는 행동으로 생각하고, 모든 평점 기록을 session으로 만들어 학습하는 경우와 일정 평점 이상의 영화만 session으로 학습하는 경우로 나눠서 진행했습니다.

<br/>

## 결과
평가지표는 HitRate으로 나타냈습니다.





