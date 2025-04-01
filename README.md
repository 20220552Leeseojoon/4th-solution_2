# 4th-solution_2

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 데이터 로드
data = pd.read_csv("02.야구선수연봉(투수).csv")  # CSV 파일 경로를 본인에 맞게 수정하세요.
print(data.shape)
print(data.info())
print(data.head())

# train/test 데이터셋 분리
train_data, test_data = train_test_split(data, test_size=0.2, random_state=1)

# 모델1 (모든 변수 포함)
model1 = sm.OLS.from_formula("SALARY_2018 ~ " + " + ".join(train_data.columns.difference(['SALARY_2018'])), data=train_data).fit()
print(model1.summary())

# 모델2 (특정 변수만 포함)
model2 = sm.OLS.from_formula("SALARY_2018 ~ VICTORY + DEFEAT + SAVE + WAR", data=train_data).fit()
print(model2.summary())

# 예측
pre_lm = model2.predict(test_data)
result = pd.DataFrame({'actual': test_data['SALARY_2018'], 'predict': pre_lm})
print(result)

# RMSE 계산
rmse_value = np.sqrt(mean_squared_error(test_data['SALARY_2018'], pre_lm))
print("RMSE:", rmse_value)



이 코드는 야구 선수들의 2018년 연봉(SALARY_2018)을 예측하기 위한 선형 회귀 분석을 수행하는 것입니다.

CSV 파일에서 데이터를 로드하고, 훈련 데이터(80%)와 테스트 데이터(20%)로 나눕니다.
모든 변수(VICTORY, DEFEAT, SAVE, WAR 등)를 사용하여 연봉을 예측하는 모델을 만들고, VICTORY, DEFEAT, SAVE, WAR만 사용하여 예측하는 모델을 만듭니다. 마지막으로 테스트 데이터를 통해 예측값을 계산하고, RMSE를 통해 모델의 정확도를 평가합니다.

회귀 분석 결과, model1은 모든 변수를 포함한 분석, model2는 주요 성적 변수들만 포함한 분석 결과를 보여줍니다.
