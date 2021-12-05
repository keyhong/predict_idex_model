# 01. Packages Loading
import os
import sys

from db_connector import DbConnector
from safe_query import QueryCollection

import pandas as pd

from datetime import datetime, timedelta

import warnings
warnings.filterwarnings(action='ignore')

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm

import joblib


def data_preprocessing(src_data: pd.DataFrame) -> pd.DataFrame:
    """

    Parameters
    ----------
    src : pd.DataFrame
       
    Returns
    -------
    train_df : pd.DataFrame
       기초 전처리가 끝난 데이터 프레임
    
    """

    def make_basic_variable(src_data: pd.DataFrame) -> pd.DataFrame:
        """ 02-(2) 기초 파생변수를 생성하는 함수

        Parameters
        ----------
        src_data : pd.DataFrame
           전처리 모듈을 거친 #별 1년 단위의 데이터
        
        Returns
        -------
        src_data : pd.DataFrame
           전처리 모듈을 거친 #별 1년 단위의 데이터
        
        Notes
        -----
           행렬번호-컬럼별로 그룹핑을 하여 파생변수를 추가한다"""
        
        # 기준시간으로 그룹핑
        grouped_tm = src_data.groupby(by=['mtr_no', 'stdr_tm'], as_index=False).agg(ext_tm_scr=('dcr_estn_scr', 'mean'), tm_alpha=('falpha_co', 'mean'))
        src_data = src_data.merge(grouped_tm, how='left', on=['mtr_no', 'stdr_tm'])
        
        # 요일명으로 그룹핑
        grouped_day_nm = src_data.groupby(by=['mtr_no', 'day_nm'], as_index=False).agg(ext_day_scr=('dcr_estn_scr', 'mean'), day_alpha=('falpha_co', 'mean'))
        src_data = src_data.merge(grouped_day_nm, how='left', on=['mtr_no', 'day_nm'])
        
        # 습도값으로 그룹핑
        grouped_hd_val = src_data.groupby(by=['mtr_no', 'hd_val'], as_index=False).agg(ext_hd_scr=('dcr_estn_scr', 'mean'))
        src_data = src_data.merge(grouped_hd_val, how='left', on=['mtr_no', 'hd_val'])
        
        # 기온값으로 그룹핑
        grouped_atp_val = src_data.groupby(by=['mtr_no', 'atp_val'], as_index=False).agg(ext_apt_scr=('dcr_estn_scr', 'mean'))
        src_data = src_data.merge(grouped_atp_val, how='left', on=['mtr_no', 'atp_val'])
        
        # 행렬번호로 그룹핑
        grouped_mtr_no = src_data.groupby(by='mtr_no', as_index=False).agg(ext_mtr_scr=('dcr_estn_scr', 'mean'), mtr_alpha_max=('falpha_co', 'max'), mtr_alpha_mean=('falpha_co', 'mean'))
        src_data = src_data.merge(grouped_mtr_no, how='left', on='mtr_no')

        # alpha_rate 변수 생성
        src_data['alpha_rate'] = round(src_data['falpha_co'] / src_data['mtr_alpha_max'], 2)
        
        return src_data


    def get_split_date(src_data: pd.DataFrame) -> str:
        """ 02-(2) 데이터프레임을 입력받아, 최소날짜와 최대날짜를 나누어  분할할 기준날짜를 선정하는 함수 
        
        Parameters
        ----------
        src_data : pd.DataFrame
        전처리 모듈을 거친 #별 1년 단위의 데이터
        
        Returns
        -------
        split_date : str
        1년 단위의 데이터를 절반의 날짜로 분할하는 분리 날짜
        
        Notes
        -----
        src_data 데이터의 min, max 날짜값을 #해 절반 분할 후 하루를 뺀다
        """
        
        # start_date : 일자가 가장 빠른 일자
        start_date = datetime.strptime(src_data['stdr_de'].min(), '%Y%m%d')
        
        # end_date : 일자가 가장 느린 일자  
        end_date = datetime.strptime(src_data['stdr_de'].max(), '%Y%m%d')

        # 두 날짜의 차이를 반으로 나눔
        delta = (end_date - start_date) / 2
        
        # start_date에 delta를 더하여 분리날짜를 반환
        split_date = (start_date + delta).strftime('%Y%m%d')

        return split_date


    def make_beta_variable(group_data: pd.DataFrame, beta_train: pd.DataFrame) -> pd.DataFrame:
        """ 02-(3) beta 파생변수 생성하는 함

        Parameters
        ----------
        group_data : pd.DataFrame
           파생 변수 데이터 셋
        beta_train : pd.DataFrame
           모델 학습 데이터 셋  
        
        Returns
        -------
        output_df : pd.DataFrame
           파생변수를 붙인 데이터
        
        Notes
        -----
        랜덤샘플링으로 인해 grid, stdr_tm, day_nm 조건을 전부 충족하는 데이터가 없는 경우
        grid/stdr_tm/day_nm -> grid/stdr_tm -> grid/day_nm -> grid -> admd_cd 순으로 N/A값을 채운다
        """
        
        group_data = group_data[['stdr_tm', 'mtr_no', 'admd_cd', 'day_nm', 'ext_apt_scr', 'ext_mtr_scr', 'ext_hd_scr', 'ext_tm_scr', 'ext_day_scr']]
        beta_train = beta_train[['stdr_de', 'stdr_tm', 'mtr_no', 'admd_cd', 'day_nm', 'dcr_estn_scr']]
        
        # 중복되는 전처리 과정 -> 함수로 변환하여 재사용 2021_08_09 홍규원 검토미완
        def grouping_mean_df(
            group_data: pd.DataFrame, beta_train: pd.DataFrame,
            mapping_lst: list, output_df: pd.DataFrame
            ):
        

            # 생성할 파생변수 리스트
            derived_var_lst = ['ext_apt_scr', 'ext_mtr_scr', 'ext_hd_scr', 'ext_tm_scr', 'ext_day_scr']

            # mapping_lst로 그룹핑하여 변수의 평균값  생성
            grouped_df = group_data.groupby(by=mapping_lst, as_index=False)[derived_var_lst].mean()
            

            # 그룹핑된 데이터프레임을 파생변수가 없는 행들과 결합하기
            merged_df = beta_train.merge(right=grouped_df, how='left', on=mapping_lst).fillna(-1)

            # 파생변수가 채워진 데이터와 null인 데이터 #분
            null_df = merged_df[merged_df.ext_apt_scr == -1].drop(labels=derived_var_lst, axis=1) # 다음 groupby에 활용하기 위해 groupby로 생성된 파생변수 제거
            
            not_null_df = merged_df[merged_df.ext_apt_scr != -1]  
            
            # 결측치가 없는 데이터프레임은 병합하고, 병합에 사용된 데이터프레임 변수 제거
            output_df = output_df.append(not_null_df, ignore_index=True)
            
            return output_df, null_df
            
        output_df = pd.DataFrame()
        
        output_df, null_df = grouping_mean_df(group_data, beta_train, ['mtr_no', 'stdr_tm', 'day_nm'], output_df)
        output_df, null_df = grouping_mean_df(group_data, null_df, ['mtr_no', 'stdr_tm'], output_df)
        output_df, null_df = grouping_mean_df(group_data, null_df, ['mtr_no', 'day_nm'], output_df)
        output_df, null_df = grouping_mean_df(group_data, null_df, 'mtr_no', output_df)
        output_df, null_df = grouping_mean_df(group_data, null_df, 'admd_cd', output_df)

        return output_df


    def make_alpha_variable(group_data: pd.DataFrame, alpha_train: pd.DataFrame) -> pd.DataFrame:
        """ 02-(4) alpha 파생변수 생성하는 함수

        Parameters
        ----------
        group_data : pd.DataFrmae
           파생 변수 데이터 셋
        alpha_train : pd.DataFrame
           모델 학습 데이터 셋  
        
        Returns
        -------
        DataFrame2 : pd.DataFrame
           파생변수를 붙인 데이터
        
        Notes
        -----
        랜덤샘플링으로 인해 grid, stdr_tm, day_nm 조건을 전부 충족하는 데이터가 없는 경우
        grid/stdr_tm/day_nm -> grid/tm -> grid/day_nm -> grid -> admd_cd 순으로 N/A값을 채운다
        """
        

        group_data = group_data[['stdr_tm', 'mtr_no', 'admd_cd', 'day_nm', 'tm_alpha', 'day_alpha', 'mtr_alpha_mean', 'mtr_alpha_max']] 
        alpha_train = alpha_train[['stdr_de', 'stdr_tm', 'mtr_no', 'admd_cd', 'day_nm', 'falpha_co']]

        def grouping_mean_df(
            group_data: pd.DataFrame, alpha_train: pd.DataFrame,
            mapping_lst: list, output_df: pd.DataFrame
            ) -> tuple(pd.DataFrame, pd.DataFrame):
            
            # 생성할 파생변수 리스트
            derived_var_lst = ['tm_alpha', 'day_alpha', 'mtr_alpha_mean', 'mtr_alpha_max']
            
            # mapping_lst로 그룹핑하여 변수의 평균값  생성
            grouped_df = group_data.groupby(by=mapping_lst, as_index=False)[derived_var_lst].mean()
            
            # 그룹핑된 데이터프레임을 파생변수가 없는 행들과 결합하기
            merged_df = alpha_train.merge(right=grouped_df, how='left', on=mapping_lst).fillna(-1)
            
            # 파생변수가 채워진 데이터와 null인 데이터 #분
            null_df = merged_df[merged_df.tm_alpha == -1].drop(labels=derived_var_lst, axis=1) # 다음 groupby에 활용하기 위해 groupby로 생성된 파생변수 제거

            not_null_df = merged_df[merged_df.tm_alpha != -1]
            
            # 결측치가 없는 데이터프레임은 병합하고, 병합에 사용된 데이터프레임 변수 제거
            output_df = output_df.append(not_null_df, ignore_index=True)
            
            return output_df, null_df
        
        output_df = pd.DataFrame()
        
        output_df, null_df = grouping_mean_df(group_data, alpha_train, ['mtr_no', 'stdr_tm', 'day_nm'], output_df)
        output_df, null_df = grouping_mean_df(group_data, null_df, ['mtr_no', 'stdr_tm'], output_df)
        output_df, null_df = grouping_mean_df(group_data, null_df, ['mtr_no', 'day_nm'], output_df)
        output_df, null_df = grouping_mean_df(group_data, null_df, 'mtr_no', output_df)
        output_df, null_df = grouping_mean_df(group_data, null_df, 'admd_cd', output_df)

        return output_df


    ''' 함수 실행 '''
    # src 데이터 기초전처리
    src_data = make_basic_variable(src_data)
    
    # 준간 날짜로 데이터 분리
    split_date = get_split_date(src_data)
    
    group_data = src_data.query('stdr_de < @split_date').reset_index(drop=True) # 파생변수 생성 데이터셋  
    train_data = src_data.query('stdr_de >= @split_date').reset_index(drop=True) # 모델학습 데이터셋

    # src 데이터 - train 데이터 파생변수 생성 (beta, alpha인 데이터 전처리)
    beta_train = make_beta_variable(group_data, train_data)
    alpha_train = make_alpha_variable(group_data, train_data)
    
    # 최종형태 - merge
    train_df = pd.merge(beta_train, alpha_train, how='left', on=['stdr_de', 'stdr_tm', 'mtr_no', 'admd_cd', 'day_nm'])
    train_df = train_df.sort_values(by=['stdr_de', 'stdr_tm', 'mtr_no']).reset_index(drop=True)
    
    return train_df


def make_beta_training_model(train_df) -> pd.DataFrame:
    """ 02-(5) beta(beta점수) 예측 모델

    Parameters
    ----------
    train_df: pd.DataFrame
       앞 변수생성 전처리가 끝난 merge 데이터

    Returns
    -------
    beta_log: pd.DataFrame
       beta 변수 예측모델링의 회귀계수 값을 로그로 기록
    
    Notes
    -----
    5개 Fold의 개별 폴더를 매개변수로 모델링 함수에 전달, 모델링 함수는 반복문을 돌며 로그와 모델을 전역변수에 저장 
    모델 생성 후  회귀계수 값을 비교하며 최적 모델 선정, 모델 파일 저장
    """
    
    df5 = train_df[['ext_apt_scr', 'ext_mtr_scr', 'ext_hd_scr','ext_tm_scr','ext_day_scr', 'dcr_estn_scr']]
    
    X = df5.drop(columns=['dcr_estn_scr']).values
    y = df5['dcr_estn_scr']

    # 모델 종류 : lm(Linear Regression model), rf(Random Forest), xgb(Xgboost), lgbm(LightGbm)
    # model_r_squared_score = {'lm':0, 'rf':0, 'xgb':0, 'lgbm':0}
    model_r_squared_score = {'lgbm': 0}
    
    # 각 모델의 예측값들을 저장할 전역변수 생성
    for model in model_r_squared_score.keys():        
        # k-fold model들 저장
        globals()['save_{}5'.format(model)] = {}

    # seed 및 cv 생성
    cv = KFold(n_splits=5, random_state=2019, shuffle=True)
    
    # 로그를 기록할 데이터 프레임 생성
    beta_log = pd.DataFrame()
    
    from sklearn.model_selection import ShuffleSplit

    ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=2019)

    # 각 모델을 만드는 함수 실행
    for fold, (train_idx, test_idx) in enumerate(tqdm(ss.split(X))):
        train_X, train_y = X[train_idx], y[train_idx]
        predict_X, predict_y = X[test_idx], y[test_idx]
        
        beta_log = LinearRegeression(train_X, train_y, predict_X, predict_y).model_training(beta_log)
        beta_log = RandomForestRegression(train_X, train_y, predict_X, predict_y).model_training(beta_log)
        beta_log = XGBoost(train_X, train_y, predict_X, predict_y).model_training(beta_log)
        beta_log = LGBM(train_X, train_y, predict_X, predict_y).model_training(beta_log)
        
        scores_lst = [lgbm_val_r2score]
                           
        for score, item in zip(scores_lst, model_r_squared_score.items()):
            model_r_squared_score[item[0]] = item[1] + score

        # beta_log = beta_log.append(pd.concat([beta_lm_log, beta_rf_log, beta_xgb_log, beta_lgbm_log], ignore_index=True))
        beta_log = beta_log.append(pd.concat([beta_lgbm_log], ignore_index=True))
        
    # 최적성능 모델을 선택, joblib 생성
    select_best_model('beta', model_r_squared_score)
    
    return beta_log
     
### 03. Model Training : alpha
def make_alpha_training_model(train_df: pd.DataFrame) -> pd.DataFrame:
    """ 02-(6) alpha 예측 모델

    Parameters
    ----------
    train_df: 앞 변수생성 전처리가 끝난 merge 데이터

    Returns
    -------
    alpha_log: alpha 변수 예측모델링의 회귀계수 값을 로그로 기록
    
    Notes
    -----
    5개 Fold의 개별 폴더를 매개변수로 모델링 함수에 전달, 모델링 함수는 반복문을 돌며 로그와 모델을 전역변수에 저장
    모델 생성 후  회귀계수 값을 비교하며 최적 모델 선정, 모델 파일 저장
    """
    
    df5 = train_df[['mtr_alpha_max', 'mtr_alpha_mean', 'tm_alpha', 'day_alpha', 'falpha_co',]]
    
    X = df5.drop(columns=['falpha_co']).values
    y = df5['falpha_co']

    # 모델 종류 : lm(Linear Regression model), rf(Random Forest), xgb(Xgboost), lgbm(LightGbm)
    model_r_squared_score = {'xgb': 0} # model_r_squared_score = {'lm': 0, 'rf': 0, 'xgb': 0, 'lgbm': 0}
    
    # 각 모델의 예측값들을 저장할 전역변수 생성
    for model in model_r_squared_score.keys():
        # k-fold model들 저장
        globals()['save_{}5'.format(model)] = {}
        

    # 로그를 기록할 데이터 프레임 생성
    alpha_log = pd.DataFrame()
    
    # seed 및 cv 생성
    kfold = KFold(n_splits=5, random_state=2019, shuffle=True)
       
    # 각 모델을 만드는 함수 실행
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
        
        train_X, train_y = X[train_idx], y[train_idx]
        predict_X, predict_y = X[test_idx], y[test_idx]
        
        alpha_log = LinearRegeression(train_X, train_y, predict_X, predict_y).model_training(alpha_log)
        alpha_log = RandomForestRegression(train_X, train_y, predict_X, predict_y).model_training(alpha_log)
        alpha_log = XGBoost(train_X, train_y, predict_X, predict_y).model_training(alpha_log)
        alpha_log = LGBM(train_X, train_y, predict_X, predict_y).model_training(alpha_log)        
        
        # scores_lst = [lm_val_r2score, rf_val_r2score, xgb_val_r2score, lgbm_val_r2score]
        scores_lst = [xgb_val_r2score]
                           
        for score, item in zip(scores_lst, model_r_squared_score.items()):
            model_r_squared_score[item[0]] = item[1] + score
            
        # alpha_log = alpha_log.append(pd.concat([alpha_lm_log, alpha_rf_log, alpha_xgb_log, alpha_lgbm_log], ignore_index=True))
        alpha_log = alpha_log.append(pd.concat([alpha_xgb_log], ignore_index=True))
    
    # 최적성능 모델을 선택, joblib 생성
    select_best_model('alpha', model_r_squared_score)   
    
    return alpha_log

def print_current_time() -> str:
    return f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'


class ModelFactory(object):
    """분석 모델을 만들고, 학습시키기 위해 모델의 종류를 모아 놓은 Factory 클래스.
    """
    
    def __init__(self, train_X, train_y, predict_X, predict_y):

        # 학습셋
        self.train_X = train_X
        self.train_y = train_y

        # 테스트셋
        self.predict_X = predict_X
        self.predict_y = predict_y


    def __del__(self):
        print('instance가 삭제되었다')
    
    @abstractmethod
    def model_training(self):
        pass

    def append_training_log(self, training_log: pd.DataFrame, train_score, test_score):
        training_log = training_log.append({'learning_date': 'a',
                                            'thing_nm': thing_nm,
                                            'model': self.__class__.__name__,
                                            'data_set': 'train set',
                                            'foldNumber': fold,
                                            'r2_score':round(train_score, 3)
                                           }, ignore_index=True)

        training_log = training_log.append({'learning_date': 'a',
                                            'thing_nm': thing_nm,
                                            'model': self.__class__.__name__,
                                            'data_set': 'test set',
                                            'foldNumber': fold,
                                            'r2_score':round(test_score, 3)
                                           }, ignore_index=True)
        
        return training_log



# 03-(1) Modeling : Linear Regression
class LinearRegeression(ModelFactory):
    def __init__(self, train_X, train_y, predict_X, predict_y):
        super().__init__(train_X, train_y, predict_X, predict_y)

    
    def model_training(self, training_log):
            """
            Parameters
            ----------

            Returns
            -------
            val_score/5: 모델의 검증용 데이터 회귀계수 비교를 위한 점수(5:k-fold 갯수)
            lm_log: Linear Regression 모델의 학습/검증용 데이터 회귀계수 값을 로그 기록
            """
            
            # LinearRegression 인스턴스 초기화
            lm = LinearRegression()

            # LinearRegression 모델학습 
            lm.fit(self.train_X, self.train_y)

            # LinearRegression 학습모델 저장
            save_lm5[self.fold] = lm5
            
            '''로그 기록하기'''
            train_score = r2_score(self.train_y, lm.predict(self.train_X))
            test_score = r2_score(self.predict_y, lm.predict(self.predict_X))
            
            training_log = super().append_training_log(training_log, train_score, test_score)

            return training_log
        

class RandomForestRegression(ModelFactory):
    ''' Modeling : RandomForest '''
    def __init__(self, train_X, train_y, predict_X, predict_y):
        super().__init__(train_X, train_y, predict_X, predict_y)
    
    def model_training(self, training_log):
            """
            Parameters
            ----------
            
            Returns
            -------
            val_score/5: 모델의 검증용 데이터 회귀계수 비교를 위한 점수(5:k-fold 갯수)
            rf_log: RandomForest 모델의 학습/검증용 데이터 회귀계수 값을 로그 기록
            """

            '''모델 선언 - 학습용 데이터 모델학습'''       
            rf_reg = RandomForestRegressor(max_depth=6)  # RandomForest 모델선언
            rf_reg.fit(self.train_X, self.train_y) # RandomForest 모델학습


            '''학습모델 저장'''   
            save_rf5[fold] = rf_reg                     # RandomForest 모델 저장
            
            
            '''로그 기록하기'''
            train_score = r2_score(self.train_y, rf_reg.predict(self.train_X))
            test_score = r2_score(self.predict_y, rf_reg.predict(self.predict_X))

            training_log = super().append_training_log(training_log, train_score, test_score)
            
            return training_log



class XGBoost(ModelFactory):
    
    def __init__(self, train_X, train_y, predict_X, predict_y):
        super().__init__(train_X, train_y, predict_X, predict_y)

    def model_training(self, training_log):
        """
        
        Parameters
        ----------
        
        Returns
        -------
        """


        '''모델 선언 - 학습용 데이터 모델학습 - foldtraining 정확도 측정''' 
        xgb = XGBRegressor(max_depth=6)  # xgb 모델선언
        xgb.fit(self.train_X, self.train_y)              # xgb 모델학습
        
        '''학습모델 저장'''    
        save_xgb5[fold] = xgb5           # xgb 모델 저장
        
        
        '''로그 기록하기'''
        train_score = r2_score(self.train_y, xgb.predict(self.train_X))
        test_score = r2_score(self.predict_y, xgb.predict(self.predict_X))

        training_log = super().append_training_log(training_log, train_score, test_score)
                                    
        
        return training_log
    
     
class LGBM(ModelFactory):
    def __init__(self, train_X, train_y, predict_X, predict_y):
        super().__init__(train_X, train_y, predict_X, predict_y)

    # 03-(4) Modeling : LightGbm
    def model_training(self, training_log):
        """
        
        Parameters
        ----------
        training_log: 

        Returns
        -------
        """
            
        '''모델 선언 - 학습용 데이터 모델학습'''         
        lgbm = lightgbm.LGBMRegressor(num_leaves=64)  # lgb 모델선언
        lgbm.fit(self.train_X, self.train_y)          # lgb 모델학습

        '''학습모델 저장'''       
        # save_lgbm5[fold] = lgbm5                     # lgb 모델 저장

        '''로그 기록하기'''
        train_score = r2_score(self.train_y, lgbm.predict(self.train_X))
        test_score = r2_score(self.predict_y, lgbm.predict(self.predict_X))

        training_log = super().append_training_log(training_log, train_score, test_score)
    
        return training_log

### 04. Best Model Selection
def select_best_model(type, model_r_squared_score):
    global save_lm5, save_rf5, save_xgb5, save_lgbm5
    
    # value의 값에 따라 키 순서를 내림차순으로 정렬
    model_r_squared_score = sorted(model_r_squared_score.items(), key=lambda x: x[1], reverse=True)
    
    # r2_score가 가장 높은 모델 선택
    highest_score_model = model_r_squared_score[0][0]
   
    '''
    if highest_score_model == 'lm':
        best_model = save_lm5
    
    elif highest_score_model == 'rf':
        best_model = save_rf5
    '''

    if highest_score_model == 'xgb':
        best_model = save_xgb5
        
    elif highest_score_model == 'lgbm':
        best_model = save_lgbm5

    '''
    save_lm5 = {}    # if LinearRegression has highest score
    save_rf5 = {}    # if RandomForest has highest score
    save_xgb5 = {}   # if XGBoost has highest score
    save_lgbm5 = {}  # if LGBM has highest score
    '''    
    
    # 최적 학습모델 저장
    
    for idx in range(len(best_model)):
        date = datetime.now().strftime('%Y%m%d')
        joblib.dump(
            value=best_model[idx],
            filename=f'./Output/Models/{thing_nm}/{type}_model_{thing_nm.lower()}_{date}_{idx}fold.joblib',
            compress=3
        )
        

def data_import(thing_cd: int):
    """쿼리 객체를 생성하고 스파크로 전처리 된 데이터를 불러오는 함수
    
    Parameters
    ----------
    thing_cd : int
       ## 코드
    src_data : pd.DataFrame
    """


    # QueryCollection 객체 생성
    query_obj = QueryCollection(today, thing_cd)

    # 전처리 전처리 데이터 로딩
    try:
        src_data = spark_session.sql(query_obj.get_dm_npa_dcr_scr_falpha_sql('###')).toPandas()
    except pyspark.sql.utils.ParseException as e:
        print(f'ParseException : {e}')
    except py4j.protocol.Py4JJavaError as e:
        print(f'Py4JJavaError : {e}')

    return src_data

### 코드 실행 ###
if __name__ == '__main__':

    print(f'--------------------프로그램 최초 시작 -- time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    # .py 실행 파라미터로 첫번째 인자에 실행현재 날짜가 들어옴
    today = str(sys.argv[1])

    # spark_session 객체 생성
    spark_session = DbConnector().create_spark_session()

    # 공간단위 thing_dict
    thing_dict = {
        '####1': 00000,
        '####2': 00000,
        '####3': 00000
    }
    

    for thing_nm, thing_cd in thing_dict.items():

        print(f'--------------------{thing_nm}: {thing_cd} 시작 -- time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        '''데이터 불러오기'''
        src_data = data_import(thing_cd)

        '''데이터 전처리'''
        src_data = data_preprocessing(src_data)
        print(f'--------------------data_preprocessing() 종료 -- time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

        ''' beta 모델학습'''
        beta_log = make_beta_training_model(src_data)
        print(f'--------------------make_beta_training_model() 종료 -- time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        ''' alpha 모델학습'''
        alpha_log = make_alpha_training_model(src_data)
        print(f'--------------------make_alpha_training_model() 종료 -- time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            
        ''' Log 결합'''
        alpha_log_df = alpha_log_df.append(alpha_log, ignore_index=True)
        beta_log_df = beta_log_df.append(beta_log, ignore_index=True)
        
        print(f'--------------------{thing_cd} 종료 -- time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    
    else:

        # 로그 데이터 프레임 행 정렬
        alpha_log_df = alpha_log_df.sort_values(by=['thing_nm', 'model', 'data_set', 'foldNumber'])
        beta_log_df = beta_log_df.sort_values(by=['thing_nm', 'model', 'data_set', 'foldNumber'])
        
        # 로그 데이터 프레임 열 정렬
        alpha_log_df = alpha_log_df[['learning_date', 'thing_nm', 'model', 'data_set' ,'foldNumber', 'r2_score']]
        beta_log_df = beta_log_df[['learning_date', 'thing_nm', 'model', 'data_set' ,'foldNumber', 'r2_score']]
        
        # 로그 데이터 파일 추출
        date = datetime.now().strftime('%Y%m%d')
        
        alpha_log_df.to_csv(f'./Output/Logs/alpha_log_{date}.csv', index=False)
        beta_log_df.to_csv(f'./Output/Logs/beta_log/beta_log_{date}.csv', index=False)
        
        print(f'--------------------프로그램 최종 종료 -- time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')