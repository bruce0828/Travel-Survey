
# coding: utf-8

# ### Travel mode detection based on the results of anchor identification

# In[1]:


import numpy as np
import pandas as pd
import os
import datetime
import pyproj
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from scipy.stats import mode
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
pd.set_option('max_rows',300)
np.set_printoptions(suppress=True)


# In[202]:


class Data(object):
    def __init__(self):
        self.base_path = 'D:\\ZhpyFile\\Travel_Survey_Analysis'
        self.pred_path = os.path.join(self.base_path,'Trip_Segment_in_pred')
        self.log_path = os.path.join(self.base_path,'Trip_Segment_in_log')
        self.mode_match = os.path.join(self.base_path,'mode_match.csv')
        self.pred_files = os.listdir(self.pred_path)
        self.log_files = os.listdir(self.log_path)
        
    def pred_dates(self):
        # all dates in pred file
        pdate = set(map(lambda x: x.split('_')[0],self.pred_files))
        return pdate
    
    def log_dates(self):
        # all dates in log file
        ldate = set(map(lambda x: x.split('_')[0],self.log_files))
        return ldate
    
    def unrecord_files(self):  
        # several trajectories that not recored in logs
        pdate = self.pred_dates()
        ldate = self.log_dates()
        return pdate - ldate

    def get_pred_file_mode(self):
        # get the pred files and modes which storaged in mode_match.csv
        # mode_match.csv lists travel modes of each original file
        try:
            return pd.read_csv(self.mode_match).dropna()[['pred_file','adjust']].values.tolist() # list, [file_name, travel_mode]
        except FileNotFoundError:
            print('The file has been moved.')

    def random_split_data(self, test_size):
        # randomly split pred data into two parts for training and test
        files, modes = zip(*self.get_pred_file_mode())
        train_data,test_data,train_mode,test_mode = train_test_split(files,modes,shuffle=True,test_size=test_size,random_state=1)
        
        # print the number
        for mode in set(train_mode):
            print(mode.ljust(9, ' '),str(train_mode.count(mode)).ljust(3,' '),test_mode.count(mode))
        print('\n')
        return sorted(train_data),sorted(test_data)
    
    def save_dataset(self,data_set,data_name='train'):
        fpath = os.path.join(self.base_path,'Dataset')
        if not os.path.exists(fpath): os.makedirs(fpath)
        with open(fpath +'\\'+ data_name+'.txt','w') as fw:
            for element in data_set:
                fw.write(element+'-')

    def read_dataset(self,data_name='train'):
        fpath = os.path.join(self.base_path,'Dataset',data_name+'.txt')
        with open(fpath,'r') as fr:
            data_set = fr.read()
        data_set = data_set[:-1].split('-')
        return data_set
    
    #------------------------------------------------------------------------------
    # 根据Satish教授建议，分时段
    def get_peak_data(self,test_size):
        # 提取文件中的早晚高峰信息，对早晚高峰分别建立数据集
        try:
            mode_match = pd.read_csv(self.mode_match)
            mode_match.dropna(inplace=True)
            mode_match.reset_index(drop=True,inplace=True)
            
            AM = mode_match[mode_match.peak_or_not == 'AM-Peak']
            PM = mode_match[mode_match.peak_or_not == 'PM-Peak']
            off = mode_match[mode_match.peak_or_not == 'off-Peak']
            
            AM_train, AM_test = train_test_split(AM[['pred_file','adjust']].values.tolist(),shuffle=True,test_size=test_size,random_state=1)
            PM_train, PM_test = train_test_split(PM[['pred_file','adjust']].values.tolist(),shuffle=True,test_size=test_size,random_state=1)
            off_train, off_test = train_test_split(off[['pred_file','adjust']].values.tolist(),shuffle=True,test_size=test_size,random_state=1)
            
            AM_train_file, AM_test_file = list(zip(*AM_train))[0], list(zip(*AM_test))[0]
            PM_train_file, PM_test_file = list(zip(*PM_train))[0], list(zip(*PM_test))[0]
            off_train_file, off_test_file = list(zip(*off_train))[0], list(zip(*off_test))[0]
            return sorted(AM_train_file),sorted(AM_test_file),sorted(PM_train_file),sorted(PM_test_file),sorted(off_train_file),sorted(off_test_file)
        except FileNotFoundError:
            print('The file has been moved.')


# In[3]:


class Preprocess(object):
    def __init__(self):
        self.data_path = 'D:\\ZhpyFile\\Travel_Survey_Analysis\\Trip_Segment_in_pred'

    def read_track(self,file):
        self.file = file
        cols = ['datetime','lon','lat','signal','true_mode','movement']
        with open(os.path.join(self.data_path,file)) as f:
            df = pd.read_csv(f,header=None,names=cols,usecols=['datetime','lon','lat','signal','true_mode'])
        df = df[df.lat<40]  # filter some noises whose lats are over 40
        df.drop_duplicates(inplace=True)
        df.datetime = df.datetime.apply(pd.to_datetime)
        df.sort_values('datetime',inplace=True)
        df['true_mode'] = df['true_mode'].apply(lambda r: self.transform_mode(r))
        df['true_mode'] = df['true_mode'].apply(lambda r: self.transform_mode_number(r))
        return df
    
    def fill_missing(self,df):
        # input: df --- read file
        length = len(df)
        duration = (df.loc[length-1,'datetime']-df.loc[0,'datetime']).seconds+1
        if length == duration:
#             print('{}:\t轨迹数据完整无缺失.'.format(self.file))
            df['missing'] = 0
        else:
#             print('{};\t轨迹数据缺失,缺失比例 {}，插值补全.'.format(self.file,length/duration))

            df.set_index('datetime',inplace = True)
            all_time = pd.date_range(df.index.min(),df.index.max(),freq='s')
            df = df.reindex(all_time)
            df['missing'] = df.lon.apply(lambda x: 1 if np.isnan(x) else 0)
            df.reset_index(inplace=True)
            df.rename(columns={'index':'datetime'},inplace=True)
            mark = df.missing.diff().fillna(0)
            marks = mark[mark!=0].index.tolist()
            starts = (np.array(mark[mark==1].index)-1).tolist()
            ends = mark[mark==-1].index.tolist()
            missing_num = len(starts)
            for i in range(missing_num):
                df.loc[starts[i]:ends[i],'lon'] = np.linspace(df.loc[starts[i],'lon'],df.loc[ends[i],'lon'],ends[i]-starts[i]+1)
                df.loc[starts[i]:ends[i],'lat'] = np.linspace(df.loc[starts[i],'lat'],df.loc[ends[i],'lat'],ends[i]-starts[i]+1)
            df.signal.fillna(0,inplace=True)
            df.drop('missing',axis=1,inplace=True)
        return df

    def utm_convert(self,LON,LAT,ZONE):
        p = pyproj.Proj(proj='utm',zone=ZONE, ellps='WGS84')  
        xx, yy =p(LON,LAT)
        return [xx, yy]

    def features(self,df):
        # velocity, acceleration, heading change rate
        df['time'] = df.datetime.apply(lambda r: datetime.time.strftime(r.time(),'%H:%M:%S'))
        df[['x','y']] = df[['lon','lat']].apply(lambda r: self.utm_convert(r.lon,r.lat,ZONE=51),axis=1)  # Shanghai ZONE=51
        df['dist'] = (df[['x','y']].diff().fillna(0)**2).sum(axis=1)**0.5
        df['speed'] = df['dist'].copy()
        df['accelerate'] = df['dist'].diff().fillna(0)
        df['jerk'] = df['accelerate'].diff().fillna(0)
        df['bearing'] = pd.Series(np.arctan((df.y.diff().fillna(0)/df.x.diff().fillna(1)).values))
        df['bear_rate'] = df['bearing'].diff().fillna(0)*180/math.pi
        df.drop('bearing',axis=1,inplace=True)
        return df
    
    def transform_mode(self,mode):
        if mode in ['eating','entertainment','home','other','school','shopping','work','visit','study','station']: return 'static'
        elif mode == 'unrecord': return 'car'         
        else: return mode
    
    def transform_mode_number(self,mode):
        # 将各种活动、方式统一为0~5数字
        if mode == 'static': return 0
        elif mode == 'walk': return 1
        elif mode == 'bicycle': return 2
        elif mode == 'bus': return 3
        elif mode in ['car','taxi']: return 4
        elif mode == 'subway': return 5
        else: print('The mode of {} is not exist.'.format(mode))


# In[4]:


class Segmentation(object):
    # split trajectory into fix_length parts
    def __init__(self,step_size,start_point,overlap_rate):
        self.step_size = step_size
        self.start_point = start_point
        self.overlap_rate = overlap_rate
        
    def split_into_segments(self,df):
        # confirm the index of dataframe start at 0
        df.reset_index(drop=True,inplace=True)
        
        start = self.start_point
        step = self.step_size
        overlap = self.overlap_rate * self.step_size
        lens = df.index.max()
        segments = dict()
        for i in range(lens+1):
            if lens - (start+(step-overlap)*i+step-1) >= 0:
                segments[i] = df.loc[start+(step-overlap)*i:start+(step-overlap)*i+step-1]
            elif lens - (start+(step-overlap)*i+step-1) < 0 and lens - (start+(step-overlap)*i+step-1) >= step/2:
                segments[i] = df.loc[start+(step-overlap)*i:len(df)]
            else:
                break
        return segments
    
    def segments_features(self,panel):
        # calculate features of segmental dataframe (followed by the split)
        dist = []
        speed_15,speed_95,speed_mean = [],[],[]
        acc_15,acc_95,acc_mean = [],[],[]
        jerk_15,jerk_95,jerk_mean = [],[],[]
        head_15,head_95,head_mean = [],[],[]
        signal = []
        trueMode = []
        for key in sorted(panel.keys()):
            dist.append(panel[key].dist.sum())
            speed_15.append(panel[key].speed.quantile(0.15))
            speed_95.append(panel[key].speed.quantile(0.95))
            speed_mean.append(panel[key].speed.mean())
            acc_15.append(panel[key].accelerate.abs().quantile(0.15))
            acc_95.append(panel[key].accelerate.abs().quantile(0.95))
            acc_mean.append(panel[key].accelerate.abs().mean())   
            jerk_15.append(panel[key].jerk.abs().quantile(0.15))
            jerk_95.append(panel[key].jerk.abs().quantile(0.95))
            jerk_mean.append(panel[key].jerk.abs().mean()) 
            head_15.append(panel[key].bear_rate.abs().quantile(0.15))
            head_95.append(panel[key].bear_rate.abs().quantile(0.95))
            head_mean.append(panel[key].bear_rate.abs().mean())      
            signal.append(panel[key].signal.sum()/len(panel[key]))
            trueMode.append(panel[key].true_mode.mode().values.tolist()[0])
            
        xx = [dist,speed_15,speed_95,speed_mean,acc_15,acc_95,acc_mean,jerk_15,jerk_95,jerk_mean,head_15,head_95,head_mean,signal]
        xx = np.array(xx).T
        yy = trueMode
        yy = np.array(yy)
        return xx,yy


# In[5]:


class Features(object):   
    def __init__(self,ss):
        self.pp = Preprocess()
        self.ss = ss
        
    def data_xy_file(self,file):
        # 对于单个文件，输出x、y矩阵
        track = self.pp.read_track(file)  # pp 读取txt
        track = self.pp.fill_missing(track)  # pp 填充缺失值
        features = self.pp.features(track)  # pp 获得整条轨迹特征
        segments = self.ss.split_into_segments(features)  # ss 分段
        xx,yy = self.ss.segments_features(segments)  # 得到片段的特征，14个
        l = len(segments)
        return xx,yy,l
        
    def data_xy_set(self,train_set):
        # derive x and y from file set
        X = np.zeros((0,14))   # 14 features
        Y = np.zeros(0)   # travel mode
        l = np.zeros(0)   # 每条轨迹划分的片段数目
        for data in train_set:
            xx,yy,ll = self.data_xy_file(data)
            X = np.r_[X,xx]
            Y = np.r_[Y,yy]
            l = np.r_[l,ll]
#         X = self.scale(X)
        return X,Y,l
    


# In[6]:


class Model_Prediciton(object):
    def __init__(self,model,mm,ss):
        self.model = model
        self.mm = mm
        self.ff = Features(ss)
    
    def predict_file(self,test_file):      
        # for single file in test set
        x_test, y_test, l_seq = self.ff.data_xy_file(test_file)
        y_pred = self.model.predict(x_test).astype(int)
        return y_pred, y_test
    
    def predict_set(self,test_set):
        # predict all test files
        y_pred, y_test = [], []
        for test_file in test_set:
            y_p, y_t = self.predict_file(test_file)
            y_pred.append(y_p.tolist())
            y_test.append(y_t.tolist())
        return y_pred, y_test


# In[7]:


class Merge(object):
    def __init__(self,MinNum):
        self.MinNum = MinNum        
        
    def merge_consecutive_mode(self,s):
        # 对某个序列，合并连续相同标签的片段
        # 输入：s--y_pred 或者 二维矩阵[mode,weight]
        # 输出：m--二维矩阵[mode,weight]，相邻mode已合并
        if s.ndim == 1:
            s = np.c_[s,np.ones_like(s),np.ones_like(s)] # 若输入为一维的y_pred，将其转化为二维数列

        m = []
        y = s[0,0]  # 当前mode标签
        w = s[0,1]  # 权重
        l = s[0,2] # 标签
        i = 1       # 序号
        t = 1       # 计数
        while i < len(s):
            if s[i,0] == y:
                t = t + 1
                w = w + s[i,1]
                i = i + 1
            else:
                m.append([y,w,l])
                t = 1
                i = i + t
                y = s[i-1,0]
                w = s[i-1,1]
                l = s[i-1,2]
        m.append([y,w,l])
        m = np.array(m)
        return m

    def search_and_combine(self,n):
        # 根据权重从大到小搜索
        # 搜索最大权重标签相邻的权重小于MinNum的元素，并合并
        # 输入：n--merge_consecutive_mode处理后的结果；MinNum--单独方式的最小长度
        # 输出：n--已搜索相邻的微片段并合并
        p = np.argmax(n[:,1]*n[:,2],axis=0)  # 指针
        y_p = n[p][0]   # 指针对应的mode值
        index = []
        if 0<p<len(n)-1:
            for i in range(p+1,len(n)):
                if n[i][1] < self.MinNum:
                    index.append(i)
                else:
                    break

            for j in range(p-1,-1,-1):   # 从p-1到0
                if n[j][1] < self.MinNum:
                    index.append(j)
                else:
                    break
                    
        elif p == 0:
            for i in range(p+1,len(n)):
                if n[i][1] < self.MinNum:
                    index.append(i)
                else:
                    break

        elif p == len(n)-1:
            for j in range(p-1,-1,-1):   # 从p-1到0
                if n[j][1] < self.MinNum:
                    index.append(j)
                else:
                    break

        n[p,1] = n[p,1] + n[index,1].sum()
        n = np.delete(n,index,axis=0)  # 三列
        n = self.merge_consecutive_mode(n)
        return n

    def loop_merge(self,q):
        # 循环，直到所有的合并可能全部完成

        i = 0
        l = len(q)
        while i < 1:
            q = self.search_and_combine(q)
            if len(q) < l:
                l = len(q)
                i = 0
            else:
                i = 1
        max_index = np.argmax(q[:,1]*q[:,2],axis=0)
        q[max_index,2] = 0
        return q

    def main_merge(self,y_pred):
        # 合并主程序, Y_pred - numpy
        m = self.merge_consecutive_mode(y_pred)
        q = self.search_and_combine(m)
        q = self.loop_merge(q)
        i = 0
        l = len(q)
        while i < 1 or q[:,2].max() != 0:
            q = self.search_and_combine(q)
            q = self.loop_merge(q)
            if len(q) < l:
                l = len(q)
                i = 0
            else:
                i = 1
        q = q[:,:2]
        q = self.flatten_merge_output(q)  # 将权重的结果展开，以便于画图
        return q
    
    def flatten_merge_output(self,mo): 
        # 将合并的结果根据权重展开，例如[[1,3],[2,4]] --> [1,1,1,2,2,2,2]
        # mo -- merge_output
        x = []
        for i in range(len(mo)):
            for j in range(mo[i,1]):
                x.append(mo[i,0])
        x = np.array(x)
        return x
    
    def delete_zeros(self,y):
        # 将合并的矩阵中0元素删除
        mo = self.merge_consecutive_mode(y)
        zero_index = []
        for i in range(len(mo)):
            if mo[i,0] == 0:
                zero_index.append(i)
        return np.delete(mo,zero_index,axis=0)[:,0]
    
    def delete_012(self,y):
        # 对于混合交通方式，去除{静止，步行，自行车}之后的交通方式
        mo = self.merge_consecutive_mode(y)
        zero_index = []
        for i in range(len(mo)):
            if mo[i,0] == 0 or mo[i,0] ==1 or mo[i,0] ==2:
                zero_index.append(i)
        return np.delete(mo,zero_index,axis=0)[:,0]


# In[45]:


class Discrete_Hidden_Markov_Model(object):
    def __init__(self,ss, x_cluster):
        self.x_cluster = x_cluster
        self.pp = Preprocess()
        self.ss = ss

    def scale(self,x):
        # 标准化，将矩阵x转化为0-1区间数值
        # input: x matrix
        return (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0))
    
    def mu_std(self,x):
        self.mu = x_train.mean(axis=0)
        self.std = x_train.std(axis=0) 
    
    def y_to_sequence(self,y):
        # 将一条轨迹的所有y值转换为序列对
        # input: y, array, single trajectory, [0,4,4,,..,4,4,4]; output: ['04','44',...,'44','44']
        y = y.astype(int)
        seq = []
        for i in range(len(y)-1):
            seq.append(str(y[i]) + str(y[i+1]))
        seq.append(str(y[len(y)-2]) + str(y[len(y)-1]))
        return seq
    
    def y_set_to_sequence(self, y_train, l_seq):
        # 将训练集中的序列转变为状态转移，便于统计
        # 由于ML将所有训练集的数据合并到一个list中，现在需要根据每段长度将其打断,因此增加了 l_seq 参数
        # input: y_train, all trajectories [0,4,4,,..,1,0,0]; output: ['04','44',...,'10','00']
        y_seq = []
        index = np.cumsum(l_seq).astype(int).tolist()
        index.insert(0,0)  # 以0开始的len(l_seq) + 1长度的list
        for i in range(len(index)-1):
            y = y_train[index[i]:index[i+1]]
            seq = self.y_to_sequence(y)
            y_seq += seq
        return y_seq
                        
    def vector_quantization_x(self,x_train):
        # 根据x_train 将连续向量转换为离散类别
        # input: [v1,v2,...,v14];  output:[label1,...,labeln], [center1,...,centern]
        self.mu_std(x_train)
        x_train = (x_train-self.mu) / (self.std+10e-10)  # 归一化
        kmeans = KMeans(n_clusters=self.x_cluster).fit(x_train)
        self.label = kmeans.labels_
        self.center = kmeans.cluster_centers_
        return self.label,self.center
        
    def transition_probability(self,y_seq):
        # 统计y_seq list 中各种转移的比例
        # input:['04','44',...,'10','00'];  output:matrix
        transit = np.zeros((6,6))  # T,W,C,B,A,S
        for i in range(6):
            for j in range(6):
                transit[i,j] = y_seq.count(str(i)+str(j))
        A = transit / transit.sum(axis=1)
        return A
        
    def emission_probability(self, y_train):
        # 根据真实的序列以及转变的观测值，求每一种序列对应各观测值的频率
        # input: [0,...,5], [0,1,...,9]
        combine = np.array(list(zip(y_train.astype(int),self.label)))
        symbol = np.zeros((6,self.x_cluster))    # 序列数*观测数
        for i in range(6):
            for j in range(self.x_cluster):
                symbol[i,j] = len(combine[(combine[:,0]==i) & (combine[:,1]==j)])
        B = symbol/symbol.sum(axis=1)[:,None]    
        return B
    
    def start_probability(self,y_train,l_seq):
        # 初始状态
        # input: [0,4,4,...] 全部
        y_train = y_train.astype(int)
        index = np.cumsum(l_seq).astype(int).tolist()
        index.insert(0,0)
        init_state = []
        for i in index[:-1]:
            init_state.append(y_train[i])
        # 统计数目
        init_num = np.zeros(6)
        for i in range(6):
            init_num[i] = init_state.count(i)
        PI = init_num / init_num.sum()
        return PI
        
    def learning(self,x_train,y_train,l_seq):
        # 训练，得到模型lambda
        y_seq = self.y_set_to_sequence(y_train, l_seq)
        label,center = self.vector_quantization_x(x_train)
        A = self.transition_probability(y_seq)
        B = self.emission_probability(y_train)
        PI = self.start_probability(y_train,l_seq)
        return A, B, PI
    
    def vector_quantization_y(self, x_test):
        # 对于一个新的x_test，求其类别；即将观测的向量转换为观测的离散类别
        # input:[x1,...,x14];  output:[label]
        dist = np.power(self.center-x_test,2).sum(axis=1)
        label_x = np.argmin(dist)
        return label_x       
    
    def trajectory_to_xlabel_ytest(self,test_file):
        # 对于测试集中一条轨迹，将所有片段的特征向量转换为类别标签
        track = self.pp.read_track(test_file)  # pp 读取txt
        track = self.pp.fill_missing(track)  # pp 填充缺失值
        features = self.pp.features(track)  # pp 获得整条轨迹特征
        segments = self.ss.split_into_segments(features)  # ss 分段
        xx,yy = self.ss.segments_features(segments)  # 得到片段的特征，14个
        xx = (xx-self.mu) / (self.std+1e-6)
        x_test = []
        for i in range(len(xx)):
            x_label = self.vector_quantization_y(xx[i])
            x_test.append(x_label)
        y_test = yy
        return np.array(x_test),y_test    
        
    def viterbit_algorithm(self,x_test,A,B,PI):
        # 预测
        obs = [str(OBS) for OBS in x_test]
        states = [str(i) for i in range(6)]
        s_pro = {str(i):PI[i] for i in range(6)}
        t_pro = {}
        for i in range(6):
            temp = {}
            t_pro[str(i)] = temp
            for j in range(6):
                temp[str(j)] = A[i,j]
        e_pro = {}
        for i in range(6):
            temp = {}
            e_pro[str(i)] = temp
            for j in range(self.x_cluster):
                temp[str(j)] = B[i,j] 
        
        path = {s:[] for s in states} # init path: path[s] represents the path ends with s
        curr_pro = {}
        for s in states:
            curr_pro[s] = s_pro[s]*e_pro[s][obs[0]]
        for i in range(1, len(obs)):
            last_pro = curr_pro
            curr_pro = {}
            for curr_state in states:
                max_pro, last_sta = max(((last_pro[last_state]*t_pro[last_state][curr_state]*e_pro[curr_state][obs[i]],                                          last_state) 
                                           for last_state in states))
                curr_pro[curr_state] = max_pro
                path[curr_state].append(last_sta)

        # find the final largest probability
        max_pro = -1
        max_path = None
        for s in states:
            path[s].append(s)
            if curr_pro[s] > max_pro:
                max_path = path[s]
                max_pro = curr_pro[s]
        return max_path
 
    def recognation_file(self,A,B,PI,test_file):
        # 对于单条轨迹，预测
        x_test, y_test = self.trajectory_to_xlabel_ytest(test_file)
        max_path = self.viterbit_algorithm(x_test,A,B,PI)
        return max_path, y_test
        
    def recognation_set(self,A,B,PI,test_set):
        # 对于所有的轨迹
        y_pred, y_test = [], []
        for test_file in test_set:
            y_p, y_t = self.recognation_file(A,B,PI,test_file)
            y_p = np.array(y_p).astype(int).tolist()
            y_pred.append(y_p)
            y_test.append(y_t)
        return y_pred, y_test


# In[114]:


class Analyze_Result():
    def __init__(self,mm):
        self.mm = mm
    
    def results(self,y_pred,y_test):
        MM = Merge(MinNum=4)  # 用于处理y_test,因为对于y_test，4为最优参数
        res = pd.DataFrame([np.array(y_pred),np.array(y_test)]).T
        res.columns = ['y_pred','y_test']
        res['yp_smth'] = res.y_pred.apply(lambda l: mm.main_merge(np.array(l)))
        res['yt_smth'] = res.y_test.apply(lambda l: MM.main_merge(np.array(l)))
        res['num']     = res.y_pred.apply(lambda l: len(l))
        res['yp_set']  = res.yp_smth.apply(lambda l: mm.delete_zeros(np.array(l)))
        res['yt_set']  = res.yt_smth.apply(lambda l: mm.delete_zeros(np.array(l)))
        res['state']   = res.yt_set.apply(lambda l: 'single' if len(l)<=1 else 'multiple')
        return res
    
    def confusion_matrix_segments(self,y_pred,y_test):
        # 对 y_pred 与 y_test 做混合矩阵
        # 输出：方式 [T W C B A S] 6*6矩阵
        pred = []
        test = []
        for i in range(len(y_pred)):
            for j in range(len(y_pred[i])):
                pred.append(y_pred[i][j])
                test.append(y_test[i][j])

        r = np.array([pred,test]).T
        cm = np.zeros((6,6))  # confusion matrix
        for i in range(6):
            for j in range(6):
                cm[i,j] = len(r[(r[:,0]==i) & (r[:,1]==j)])
        return cm
    
    def precision(self,y_pred,y_test):
        # 返回各种方式的准确率
        cm = self.confusion_matrix_segments(y_pred,y_test)
        return np.diag(cm) / cm.sum(axis=1)
    
    def recall(self,y_pred,y_test):
        # 返回各种方式的召回
        cm = self.confusion_matrix_segments(y_pred,y_test)
        return np.diag(cm) / cm.sum(axis=0)
    
    def accuracy(self,y_pred,y_test):
        cm = self.confusion_matrix_segments(y_pred,y_test)
        return np.diag(cm).sum() / cm.sum().sum()
    
    def main_evaluate(self,y_pred,y_test):
        cm = self.confusion_matrix_segments(y_pred,y_test)
        p = self.precision(cm)
        r = self.recall(cm)
        a = self.accuracy(cm)
        return cm, p, r, a
    
    def display(self, y_pred, y_smooth, y_test, filename):
        fig = plt.figure(figsize=(12,1.2))
        n = len(y_pred)
        # mode_mapping = {'static': 0,'walk': 1,'bicycle': 2,'bus': 3,'car': 4,'subway': 5}

        def colors(R,G,B): return "#%02X%02X%02X" % (R,G,B)



        mc = [colors(171,132,191),    # 灰色 static     # mc = mode_color
              colors(243,195,37),  # 黄色 walk
              colors(148,198,39),    # 绿色 bicycle
              colors(5,154,71),  # 青色 bus
              colors(236,98,0),  # 棕色 car
              colors(45,111,171)]  # 蓝色 subway

        mn = ['T','W','C','B','A','S']  # mn = mode_name, stands for static, walk, cycle, bus, automobile, subway, respectively

        left = 0     # left alignment of data starts at zero
        y1 = 1
        y2 = 1.1
        y3 = 1.2

        for i in range(n):
            plt.barh(y1, width=10,height=0.05, color=mc[int(y_pred[i])], align='center', left=left, tick_label=int(y_pred[i]))
            plt.barh(y2, width=10,height=0.05, color=mc[int(y_smooth[i])], align='center', left=left, tick_label=int(y_smooth[i]))
            plt.barh(y3, width=10,height=0.05, color=mc[int(y_test[i])], align='center', left=left, tick_label=int(y_test[i]))
            left += 10
        #         plt.text(5+10*i,y1, "%s" % (mn[int(y_pred[i])]), {'color': 'w', 'fontsize': 10, 'ha': 'center', 'va': 'center',})    
        #         plt.text(5+10*i,y2, "%s" % (mn[int(y_smooth[i])]), {'color': 'w', 'fontsize': 10, 'ha': 'center', 'va': 'center',})
        #         plt.text(5+10*i,y3, "%s" % (mn[int(y_test[i])]), {'color': 'w', 'fontsize': 10, 'ha': 'center', 'va': 'center',})

        plt.yticks([y1,y2,y3],['Pre-merging','Post-merging','Ground truth'],fontsize=14)
        plt.xlim([0,10*n])
        plt.xticks([])
        # plt.xlabel('Segment sequence',fontsize=12)
        # plt.title('Results of 30-s segmentation',fontsize=14)

        plt.savefig(dpi=600,bbox_inches='tight',fname=filename)
        ("")



# ### 主程序
# #### 分别运用HMM、RF、SVM预测

# In[10]:


# split data into two parts
dd = Data()
#     train_set, test_set = dd.random_split_data(test_size=1/3)

# save the train_set and test_set
#     dd.save_dataset(train_set,data_name='train')
#     dd.save_dataset(test_set,data_name='test')
train_set, test_set = dd.read_dataset('train'), dd.read_dataset('test') 


# In[11]:


pp = Preprocess()
ss = Segmentation(step_size=30, start_point=0, overlap_rate=0)
ff = Features(ss)  # Segmentation, model, train_set
x_train, y_train, l_seq = ff.data_xy_set(train_set)


# In[12]:


mm = Merge(MinNum=5)
ar = Analyze_Result(mm)


# In[13]:


# 隐马尔可夫模型
DHMM = Discrete_Hidden_Markov_Model(ss,x_cluster=5)
A, B, PI = DHMM.learning(x_train,y_train,l_seq)
y_pred_dhmm, y_test_dhmm = DHMM.recognation_set(A,B,PI,test_set)
res_dhmm = ar.results(y_pred_dhmm,y_test_dhmm)
acc_dhmm = ar.accuracy(res_dhmm.y_pred,res_dhmm.yt_smth)

# In[162]:

ar.accuracy(res_dhmm.yp_smth,res_dhmm.yt_smth)


# In[17]:


# 随机森林模型
RF = RandomForestClassifier(n_estimators=400,)  # model
RF = RF.fit(x_train,y_train)
mp = Model_Prediciton(RF,mm,ss)
y_pred_rf, y_test_rf = mp.predict_set(test_set)
res_rf = ar.results(y_pred_rf,y_test_rf)
acc_rf = ar.accuracy(res_rf.yp_smth,res_rf.yt_smth)


# In[23]:


# 支持向量机模型
SVM = SVC()
SVM = SVM.fit(x_train,y_train)
mp2 = Model_Prediciton(SVM,mm,ss)
y_pred_svm, y_test_svm = mp2.predict_set(test_set)
res_svm = ar.results(y_pred_svm,y_test_svm)
acc_svm = ar.accuracy(res_svm.yp_smth,res_svm.yt_smth)


# In[161]:


ar.accuracy(res_svm.y_pred,res_svm.yt_smth)


# In[16]:


# 梯度提升树模型
GBDT = GradientBoostingClassifier(n_estimators=300)
GBDT = GBDT.fit(x_train,y_train)
mp3 = Model_Prediciton(GBDT,mm,ss)
y_pred_gbdt, y_test_gbdt = mp3.predict_set(test_set)
res_gbdt = ar.results(y_pred_gbdt,y_test_gbdt)
acc_gbdt = ar.accuracy(res_gbdt.yp_smth,res_gbdt.yt_smth)


# In[174]:


# 神经网络
from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(hidden_layer_sizes=(160,))
MLP = MLP.fit(x_train,y_train)
mp4 = Model_Prediciton(MLP,mm,ss)
y_pred_mlp, y_test_mlp = mp4.predict_set(test_set)
res_mlp = ar.results(y_pred_mlp,y_test_mlp)
acc_mlp = ar.accuracy(res_mlp.yp_smth,res_mlp.yt_smth)


# In[175]:


ar.accuracy(res_mlp.y_pred,res_mlp.yt_smth)


# In[285]:


# stacking model
# res_dhmm,res_rf,res_svm,res_gbdt,res_mlp

stack = []
for i in range(len(res_rf)):
    res_matr = np.r_[res_rf.loc[i,'y_pred'],res_dhmm.loc[i,'y_pred'],res_svm.loc[i,'y_pred'], \
                     res_gbdt.loc[i,'y_pred'],res_mlp.loc[i,'y_pred']].reshape(5,-1)[0]
    stack.append([i,res_matr])

res_stack = pd.DataFrame(stack,columns=['index','y_pred'])
ar.accuracy(res_stack.y_pred,res_rf.yt_smth)


# In[290]:

# stacking
from mlxtend.classifier import StackingClassifier

stacking = StackingClassifier(classifiers=[RF,SVM,GBDT,MLP,DHMM],meta_classifier=RF)
stacking = stacking.fit(x_train,y_train)
mp_s = Model_Prediciton(stacking,mm,ss)
y_pred_stack, y_test_stack = mp_s.predict_set(test_set)
res_stack = ar.results(y_pred_stack,y_test_stack)
acc_stack = ar.accuracy(res_stack.yp_smth,res_stack.yt_smth)


# In[ ]:

# #### 根据时段划分数据集

# In[212]:


dd = Data()
AM_train,AM_test,PM_train,PM_test,off_train,off_test = dd.get_peak_data(test_size=1/3)

dd.save_dataset(AM_train,data_name='AM_train')
dd.save_dataset(AM_test,data_name='AM_test')
dd.save_dataset(PM_train,data_name='PM_train')
dd.save_dataset(PM_test,data_name='PM_test')
dd.save_dataset(off_train,data_name='off_train')
dd.save_dataset(off_test,data_name='off_test')

AM_train,AM_test = dd.read_dataset('AM_train'), dd.read_dataset('AM_test')
PM_train,PM_test = dd.read_dataset('PM_train'), dd.read_dataset('PM_test')
off_train,off_test =dd.read_dataset('off_train'), dd.read_dataset('off_test') 

# In[218]:


x_AM_train, y_AM_train, l_AM_seq = ff.data_xy_set(AM_train)
RF_AM = RandomForestClassifier(n_estimators=400,)
RF_AM = RF_AM.fit(x_AM_train, y_AM_train)
mp_AM = Model_Prediciton(RF_AM,mm,ss)
y_pred_AM, y_test_AM = mp_AM.predict_set(AM_test)
res_AM = ar.results(y_pred_AM,y_test_AM)
acc_AM = ar.accuracy(res_AM.yp_smth,res_AM.yt_smth)


# In[219]:


ar.confusion_matrix_segments(res_AM.yp_smth,res_AM.yt_smth)


# In[220]:


ar.precision(res_AM.yp_smth,res_AM.yt_smth)


# In[221]:


ar.recall(res_AM.yp_smth,res_AM.yt_smth)


# In[222]:


x_PM_train, y_PM_train, l_PM_seq = ff.data_xy_set(PM_train)
RF_PM = RandomForestClassifier(n_estimators=400,)
RF_PM = RF_PM.fit(x_PM_train, y_PM_train)
mp_PM = Model_Prediciton(RF_PM,mm,ss)
y_pred_PM, y_test_PM = mp_PM.predict_set(PM_test)
res_PM = ar.results(y_pred_PM,y_test_PM)
acc_PM = ar.accuracy(res_PM.yp_smth,res_PM.yt_smth)

# In[223]:


ar.confusion_matrix_segments(res_PM.yp_smth,res_PM.yt_smth)


# In[224]:


ar.precision(res_PM.yp_smth,res_PM.yt_smth)


# In[225]:


ar.recall(res_PM.yp_smth,res_PM.yt_smth)


# In[226]:


x_off_train, y_off_train, l_off_seq = ff.data_xy_set(off_train)
RF_off = RandomForestClassifier(n_estimators=400,)
RF_off = RF_off.fit(x_off_train, y_off_train)
mp_off = Model_Prediciton(RF_off,mm,ss)
y_pred_off, y_test_off = mp_off.predict_set(off_test)
res_off = ar.results(y_pred_off,y_test_off)
acc_off = ar.accuracy(res_off.yp_smth,res_off.yt_smth)


# In[227]:


ar.confusion_matrix_segments(res_off.yp_smth,res_off.yt_smth)


# In[228]:


ar.precision(res_off.yp_smth,res_off.yt_smth)


# In[229]:


ar.recall(res_off.yp_smth,res_off.yt_smth)

# In[ ]:


if __name__ == '__main__':
    
    t0 = datetime.datetime.now()
    # split data into two parts
    dd = Data()
#     train_set, test_set = dd.random_split_data(test_size=1/3)
    
    # save the train_set and test_set
#     dd.save_dataset(train_set,data_name='train')
#     dd.save_dataset(test_set,data_name='test')
    train_set, test_set = dd.read_dataset('train'), dd.read_dataset('test')    
    
    
    c = []
    for step_size in [1,5,10,15,30,40,50,60,80,100,120,150]:
        # segmentation
        ss = Segmentation(step_size, start_point=0, overlap=0)

        # training
        rf = RandomForestClassifier(n_estimators=500,)  # model
        tt = Trainer(ss,rf,train_set)  # Segmentation, model, train_set
        x_train, y_train = tt.data_xy(train_set)
        rf = rf.fit(x_train,y_train)


        # predict and test
        res_df = tt.predict_set(rf,test_set)
        accuracy = tt.comprehensive_accuracy(res_df)
        c.append([step_size,accuracy])
        print(step_size)
    
    print('spent time: {}'.format(datetime.datetime.now()-t0))


# 1. 判断是多方式还是单方式；
# 2. 起始位置与重合阈值
