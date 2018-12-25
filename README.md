T-Brain 旅遊訂單預測競賽
==

## EDA
這次在比賽中主要使用下面這兩種工具對特徵進行探索
 - histogram
     - 看單一特徵是否具備區辨target value的能力
     - 檢查舉辦方是否利用單一特徵進行train-test split
         - 這次比賽重啟後的testing set的劃分是根據兩點
             - begin_date在2018年之後的資料都是testing set
             - training set與testing set的group_id互相獨立
 - scatter plot: 看兩個特徵互相搭配是否具備區辨target value的能力

## Feature Engineering
 - group.csv
     - price
         - 每個旅行團在同樣sub_line或area之下的z-score
     - days
         - 同上
     - price/days
     - product_prog
         - 這次用doc2vec直接轉換成numerical feature，效果不好
 - airline.csv
     - 對機場做mean encoding，每個旅行團分別對這些mean encoding完的機場抽統計數字
         - min
         - max
         - mean
         - std
         - median
         - 這些特徵效果不好，最後沒放進去
    - 對每個旅行團的飛行時間與距離抽統計數字
        - min
        - max
        - mean
        - std
        - median
        - 這次是直接overall的抽統計數字，但是應該要對去程及回程分別抽
 - order.csv
     - 每個旅行團被下訂單的次數
     - unit處理group的次數(unit-group count)
     - unit-source_1 mean encoding
     - 每筆訂單與相同旅行團的第一筆訂單的時間差(days)
     - 第一及第二個特徵都是分別在training set與testing set上估計，這樣會造成leakage，也沒有實際應用的價值。因為traing set與testing set的group_id互相獨立，所以這個資料無法在training set上估計並mapping到testing set。用**累積次數**代替或許是一個解決方法

 - group-order
     - 下單時間與出團時間差(days)

## Model
這次比賽主要是用**catboost**做測試，最後再用stacking

### catboost
catboost與lightgbm一樣，可以直接吃categorical feature，不同的是catboost在模型內部會將categorical feature用特殊的計數方式轉成numerical feature。且因為**樹的結構**，catboost相較於其他gbm來說，比較**不容易overfitting**。
gbm調參數的流程大抵上是
 1. depth
 2. 其他與樹結構有關的參數
 3. 與隨機性有關的參數
 4. learning rate & number of iteration

### stacking
base model
 - xgboost
     - depth: 3
     - 把categorical feature當成是numerical feature

 - catboost
 - KNN: 所有categorical feature轉成mean encoding
 - logistic regression: 同上
 - 2 fully-connected neural network
     - 3 hidden layers
     - 1 hidden layers

meta model
 - logistic regression

## 總結
在這次比賽中，我的表現有下面幾個缺點
 - 整體程式架構散亂
 - EDA做得不夠到位
 - 沒有充分利用所有資料: 比如airline.csv利用得較少，而關於文字的資料更是一點都沒有利用
 - 在產生特徵的時候造成data leak

接下來可以努力的方向有
 - 多看看別人做的EDA
 - 參考一下別人的機器學習競賽pipeline都怎麼寫

通過這次比賽，我發現tune模型真的是時間及工作量占比最小的一個部分，成績能有多好，還是取決於做出來的特徵好不好。而且因為在比賽中，幾乎大家都是用gbm當做主力模型，所以最後還是比誰找得特徵更好，能夠充分的榨乾資料的價值。  
所以往後參加比賽的時候，應該將大部分的時間分配給以下三個任務
 - 真正的弄懂問題
 - EDA
 - Feature engineering

且每一次動手之前都要先充分的思考
 - 目的
 - 預期效果
 - 如何做
