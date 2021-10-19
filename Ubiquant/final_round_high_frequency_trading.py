from logging import exception
import time

import numpy as np
import random
from numpy.lib.shape_base import take_along_axis
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import sklearn.metrics as sm
import joblib

from numba import jit
 
import grpc
import question_pb2
import question_pb2_grpc
import contest_pb2
import contest_pb2_grpc
import common_pb2
import common_pb2_grpc
# import model as md
# 追求alpha拥抱beta


class client:
    user_id = xxx
    user_pin = 'xxx'
    conn_question = grpc.insecure_channel('xxx')
    conn_contest = grpc.insecure_channel('xxx')
    client_contest = contest_pb2_grpc.ContestStub(channel=conn_contest)
    client_question = question_pb2_grpc.QuestionStub(channel=conn_question)
    columns = ['xxx']
    for i in range(1,11):
        columns.append('xxx'+str(i)+'_price')
        columns.append('xxx'+str(i)+'_volume')
    for i in range(1,11):
        columns.append('xxx'+str(i)+'_price')
        columns.append('xxx'+str(i)+'_volume')
    weight = []
    for i in range(1,11):
        weight.append(0)
        weight.append(1 - (i-1)/10)
    weight = weight+weight 
    weight = np.array(weight)

    def __init__(self):
        # login
        self.session_key = None 
        self.login_success = None
        # self.login_fail_reason = None
        
        # question get
        self.sequence = 0 # 数据index,首次为0
        self.has_next_question = None # 后续是否有数据
        self.capital = 200000000 # 总资产
        self.dailystk = None # 数据！共500支股票
        self.positions = None # 当前持仓 
        
        self.model = joblib.load('/root/决赛/rf.model')
        self.refuse_list = [xxx]

        
        # submit
        self.new_position = None
        self.accepted = None
        self.max_amount = self.capital*2*0.95
        self.price = None

        # data storage
        self.count = 0
        self.all_data = pd.read_csv('/root/决赛/CONTEST_DATA_MIN_SP_1.csv',names= self.columns)
        self.new_data = None


    def login(self):
        loginRes = self.client_contest.login(contest_pb2.LoginRequest(
            user_id=self.user_id,
            user_pin=self.user_pin
            ))
        self.session_key = loginRes.session_key # 用于提交position
        self.login_success = loginRes.success # 是否成功login
        if self.login_success == False:
            print(loginRes.reason)


    def getint(self,x):
        return int(x/100)*100

    def trade(self):
        factor_list = []
        al_data = self.all_data.copy()
        al_data = al_data[(al_data['sequence']<=self.sequence)&(al_data['sequence']>self.sequence-5)]
        al_data['del'] = al_data['stock_id'].apply(lambda x: 1 if x in self.refuse_list else 0)
        al_data  = al_data[al_data['del']==0]
        grouped_cal = al_data.groupby('stock_id')
        for stock,tempstock in grouped_cal:
            # print(cal.columns)
            cal_np = np.array(tempstock.iloc[:,8:48])
            cal_np = np.row_stack((cal_np,self.weight))
            vt_b_weighted_2 = (cal_np[3]*cal_np[5])[0:20].sum()/5.5
            pt_b_weighted_2 = cal_np[3][0]
            vt_b_weighted_1 = (cal_np[4]*cal_np[5])[0:20].sum()/5.5
            pt_b_weighted_1 = cal_np[4][0]
            vt_a_weighted_2 = (cal_np[3]*cal_np[5])[20:40].sum()/5.5
            pt_a_weighted_2 = cal_np[3][20]
            vt_a_weighted_1 = (cal_np[4]*cal_np[5])[20:40].sum()/5.5
            pt_a_weighted_1 = cal_np[4][20]
            mpb_volume = tempstock['volume'].iloc[-1]
            mpb_trv = tempstock['trv'].iloc[-1]

            factor_list.append([stock,vt_b_weighted_2,pt_b_weighted_2,vt_b_weighted_1,pt_b_weighted_1,vt_a_weighted_2,pt_a_weighted_2,vt_a_weighted_1,pt_a_weighted_1,mpb_volume,mpb_trv])

        factor_df = pd.DataFrame(factor_list,columns = ['stock_id','vt_b_weighted_2','pt_b_weighted_2','vt_b_weighted_1','pt_b_weighted_1','vt_a_weighted_2','pt_a_weighted_2','vt_a_weighted_1','pt_a_weighted_1','mpb_volume','mpb_trv'])
        factor_df['delta_vt_b'] = factor_df.apply(lambda x: 0 if x['pt_b_weighted_1']<x['pt_b_weighted_2'] else x['vt_b_weighted_1'] if x['pt_b_weighted_1']>x['pt_b_weighted_2'] else x['vt_b_weighted_1']-x['vt_b_weighted_2'],axis =1)
        factor_df['delta_vt_a'] = factor_df.apply(lambda x: x['vt_a_weighted_1'] if x['pt_a_weighted_1']<x['pt_a_weighted_2'] else 0 if x['pt_a_weighted_1']>x['pt_a_weighted_2'] else x['vt_a_weighted_1']-x['vt_a_weighted_2'],axis=1)
        factor_df['dif'] = ((factor_df['pt_b_weighted_1'].values+factor_df['pt_a_weighted_1'].values)/2+(factor_df['pt_b_weighted_2'].values+factor_df['pt_a_weighted_2'].values)/2)/2
        factor_df['oir'] = (factor_df['vt_b_weighted_1'].values-factor_df['vt_a_weighted_1'].values)/(factor_df['vt_b_weighted_1'].values+factor_df['vt_a_weighted_1'].values)
        factor_df['voi2'] = factor_df['delta_vt_b'].values - factor_df['delta_vt_a'].values
        factor_df['mpb'] = factor_df['mpb_trv']/factor_df['mpb_volume']-factor_df['dif']
        df_zscore = factor_df[['stock_id','oir','voi2','mpb']].dropna().copy()

        # 训练模型
        df_zscore['pred_label'] = self.model.predict(np.array(df_zscore.iloc[:,1:4]))
        df_zscore = df_zscore.sort_values(by='pred_label',ascending=False)
        # print(df_zscore)
        df_zscore.dropna(inplace=True)

        # in case not enough pred_label
        try:
            ranker_high = df_zscore['pred_label'].iloc[49]
            ranker_low = df_zscore['pred_label'].iloc[-50]
            # df_zscore['buy_amount_1'] = df_zscore['pred_label'].apply(lambda x : self.max_amount/100 if x>=ranker_high else 0)
            # df_zscore['buy_amount_2'] = df_zscore['pred_label'].apply(lambda x : -self.max_amount/100 if x<=ranker_low else 0)
            df_zscore['buy_amount_1'] = np.where(df_zscore['pred_label']>=ranker_high,self.max_amount/100,0)
            df_zscore['buy_amount_2'] = np.where(df_zscore['pred_label']<=ranker_low,-self.max_amount/100,0)

            df_zscore['buy_amount'] = df_zscore['buy_amount_1'].values + df_zscore['buy_amount_2'].values
            df_zscore['buy_amount'] = np.vectorize(self.getint)(df_zscore['buy_amount'])
        except:
            ranker_high = 0.0005
            ranker_low = -0.0005
            # df_zscore['buy_amount_1'] = df_zscore['pred_label'].apply(lambda x : self.max_amount/100 if x>=ranker_high else 0)
            # df_zscore['buy_amount_2'] = df_zscore['pred_label'].apply(lambda x : -self.max_amount/100 if x<=ranker_low else 0)
            df_zscore['buy_amount_1'] = np.where(df_zscore['pred_label']>=ranker_high,self.max_amount/100,0)
            df_zscore['buy_amount_2'] = np.where(df_zscore['pred_label']<=ranker_low,-self.max_amount/100,0)

            df_zscore['buy_amount'] = df_zscore['buy_amount_1'].values + df_zscore['buy_amount_2'].values
            df_zscore['buy_amount'] = np.vectorize(self.getint)(df_zscore['buy_amount'])
        
        #准备输出结果
        result = self.all_data[self.all_data['sequence'] == self.sequence].copy()
        result = pd.merge(result,df_zscore,how='left',on='stock_id')
        result['buy_amount'].fillna(0,inplace=True)
        # result['buy_price'].fillna(0,inplace=True)
        result['positions'] = self.positions
        result['change'] = result['buy_amount'].values / result['close'].values - result['positions'].values
        # result['buy_price_1'] = result['pred_label'].apply(lambda x : 1 if x>=ranker_high else 0)
        # result['buy_price_2'] = result['pred_label'].apply(lambda x : 1 if x<=ranker_low else 0)
        result['buy_price_1'] = np.where(result['pred_label']>=ranker_high,1,0)
        result['buy_price_2'] = np.where(result['pred_label']<=ranker_low,1,0)
        result['buy_price'] = result['buy_price_1'].values*result['ask10_price'].values + result['buy_price_2'].values*result['bid10_price'].values
        # print(np.array(result['buy_price']))
        # print(result['buy_amount'])
        self.bidasks = np.array(result['change'])
        self.price = np.array(result['buy_price'])


    def submit(self):
        print(int(self.sequence))
        response_answer = self.client_contest.submit_answer_make(contest_pb2.AnswerMakeRequest(
            user_id=self.user_id,
            user_pin=self.user_pin,
            session_key=self.session_key, # 使用login时系统分配的key来提交
            sequence=int(self.sequence), # 使用getdata时获得的sequence
            bidasks=self.bidasks, 
            prices = self.price
        )) 
        self.accepted = response_answer.accepted # 是否打通提交通道
        if not self.accepted:
            print(response_answer.reason) # 未成功原因

    def run(self):
        self.login()
        print(f'Log in result: {self.login_success} ...')
        try:
            # while True:
            #     time.sleep(2) # 隔1秒 搞一次
            #     self.getdata()
            #     print(f'Sequence now: {self.sequence} ...')
           
            for response_question in self.client_question.get_question(question_pb2.QuestionRequest(
                user_id=self.user_id,
                user_pin=self.user_pin,
                session_key = self.session_key
            )):
                self.positions = response_question.positions # 之后的寻求数据sequence为这个sequence num + 1 
                # self.has_next_question = response_question.has_next_question # True - 后续仍有数据
                self.capital = response_question.capital # 总资产
                self.dailystk = response_question.dailystk # 数据！共500支股票
                newdata = pd.DataFrame(np.asarray([array.values for array in self.dailystk]),columns=self.columns)
                self.sequence =  newdata['sequence'].iloc[0]
                if self.count == 0:
                    self.positions = [0 for i in range(0,500)]
                    self.count += 1
                    # if newdata['sequence'].iloc[0] in list(self.all_data['sequence']):
                    #     pass
                    # else:
                    #     self.new_data = newdata
                    #     self.all_data = pd.concat([self.all_data,self.new_data])
                    self.trade()
                    self.submit()
                    print(f'Submit result: {self.accepted} ...')
                else:
                # if self.sequence == 0:
                    # print(self.positions)
                    # if newdata['sequence'].iloc[0] in list(self.all_data['sequence']):
                    #     pass
                    # else:
                    #     self.new_data = newdata
                    #     self.all_data = pd.concat([self.all_data,self.new_data])
                    if self.positions == []:
                        self.positions = [0 for i in range(0,500)]
                    time_start=time.time()
                    self.trade()
                    time_end=time.time()
                    print('totally cost',time_end-time_start)
                    
                    self.submit()
                    print(self.capital)
                    print(f'Submit result: {self.accepted} ...')
                
        except KeyboardInterrupt:
            print(self.all_data.tail(5))
            print(self.all_data.head(5))
            self.all_data.to_csv('./all_data.csv')
            return      


if __name__ == '__main__':

    new_trade = client()
    new_trade.run()

    

