import time
import numpy as np
import pandas as pd
from functools import wraps
from sklearn.model_selection import train_test_split
from collections import defaultdict
from module_aladin.data_process import concat_data

def record_time(func):
    @wraps(func) 
    def wrapper(*args,**kwargs):
        start = time.perf_counter()
        rslt = func(*args,**kwargs)
        end = time.perf_counter()
        return rslt, end-start
    return wrapper

def lists_append_together(lists:list,data:list):
    tuple(map(lambda x : x[0].append(x[1]),zip(lists,data)))
    return lists

def class_name(clss):
    name = str(type(clss)).strip()
    name = name[1:-1].split(' ')
    return name[1]

def train_test_split_strat(X:pd.DataFrame,y:pd.Series,strat=None,method='order',harsh=False,n_strata=10,**kwargs):
    # strat을 기준으로 균등하도록 train, test로 나눔
    # strat의 값을 n등분 하여 균등하게 할지, strat의 순서를 기준으로 균등하게 할지 입력 받음
    if strat is None : strat = y
    if method in ['quantile','order'] :
        p_arr = np.linspace(0,n_strata)/n_strata
        cut_p = np.quantile(strat,p_arr)
    elif method == 'value':
        cut_p = np.linspace(np.min(strat)-1,np.max(strat),n_strata)
    cut_p[-1] = np.max(strat)+1
    train_Xs,test_Xs,train_ys,test_ys = [],[],[],[]
    data = [train_Xs,test_Xs,train_ys,test_ys]
    res_Xs,res_ys =[],[]
    for p_a,p_b in zip(cut_p[:-1],cut_p[1:]):
        cond= (p_a <= strat) & (strat < p_b)
        input_X, input_y = X[cond], y[cond]
        if len(input_X) == 0 : continue
        if len(input_X) == 1 : 
            res_Xs.append(input_X), res_ys.append(input_y)
        else :
            splited = train_test_split(input_X,input_y,**kwargs)
            data = lists_append_together(data,splited)
    if len(res_Xs) > 1 :
        input_X, input_y = pd.concat(res_Xs), pd.concat(res_ys)
        splited = train_test_split(input_X,input_y,**kwargs)
        data = lists_append_together(data,splited)
    elif len(res_Xs) == 1:
        if harsh : x_ind, y_ind = 1, 3
        else : x_ind, y_ind = 0, 2
        data[x_ind],data[y_ind] = lists_append_together([data[x_ind],data[y_ind]],[res_Xs[0],res_ys[0]])
    
    return tuple(map(concat_data,data))

def get_amp(data):
  return {
      'max-min' : data['max'] - data['min'],
      'real-min' : data['real'] - data['min'],
      'avg-min' : data['avg'] - data['min'],
  }

def get_vals_range(val,decrease=True):
  moving_ds = pd.Series(val).rolling(10)
  moving = {'max' : moving_ds.max().bfill().values,
            'real': val,
            'min' : moving_ds.min().bfill().values,
            'avg' : moving_ds.mean().bfill().values}
  if decrease : lim_range = {'max':np.array([np.max(val[i:]) for i in range(len(val)-1)]),
                        'real':val[:-1],
                        'min':np.array([np.min(val[:i+1]) for i in range(len(val)-1)])}
  else : lim_range = {'max':np.array([np.max(val[:i+1]) for i in range(len(val)-1)]),
                        'real':val[:-1],
                        'min':np.array([np.min(val[i:]) for i in range(len(val)-1)])}
  lim_range['avg'] = (lim_range['max']+lim_range['min'])/2
  return moving, lim_range
  
def read_run_rslt(run_rslt):
  lines = run_rslt.strip().split('\n')
  data_line= lines[1::2]
  data = [line.strip().split('\t') for line in data_line]
  rslt = defaultdict(list)
  for line in data:
    for vals in line:
      key,val= vals.strip().split(':')
      rslt[key].append(float(val))

  return rslt