import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
import seaborn as sns
from itertools import product

import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats("png2x")
# 테마 설정: "default", "classic", "dark_background", "fivethirtyeight", "seaborn"
mpl.style.use("fivethirtyeight")
# 이미지가 레이아웃 안으로 들어오도록 함
mpl.rcParams.update({"figure.constrained_layout.use": True})

import matplotlib.font_manager as fm
#font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
#[fm.FontProperties(fname=font).get_name() for font in font_list if 'D2C' in font]
#plt.rc('font', family='D2Coding')
mpl.rcParams['axes.unicode_minus'] = False

from module_aladin.util import get_vals_range, get_amp

## FUNCTIONS - PLOTTING

def choose_split_point(word_len,space,ths):
    # 윗 줄에 space 만큼 공백이 있고, 한 줄의 길이가 ths로 제한 되어있을 때
    # 어떤 지점에서 단어를 끊어줄지 정하기
    # |-------ths-------|
    # |-space-|---------|-space-|------| : word
    #         |-------ths-------|
    print(word_len,space,ths)
    if word_len < ths + space :
        if abs(word_len/2 -ths) <= abs(word_len/2-space) :
            return word_len-ths
        else :
            return word_len - space if word_len < 2 * space else space
    else :
        return ths if word_len - (ths + space) < 0.3 * ths else space

def minimize_seq_idx_np(domain:np.array,func):
    vfunc = np.vectorize(func)
    temp = np.argsort(vfunc(domain))
    return temp[0]

def modify_strlen_ths(last,new,ths=16):
    front = len(last)
    space = ths - (1+front)
    if len(new) < space :
        rslt = [last + ' ' + new]
    else :
        if len(new) < ths:
            rslt = [last, new]
        else:
            cut = choose_split_point(len(new),space-1,ths-1)
            new_h, new_e = new[:cut]+'-', new[cut:]
            if cut < ths-1 :
                rslt = modify_strlen_ths(last+' '+new_h,new_e)
            else :
                rslt = [last] + modify_strlen_ths(new_h,new_e) 
    return rslt

def str_cutter(sentnc, ths = 16):
    words= sentnc.split(' ')
    rslt, pnt = [''], 0
    while pnt < len(words):
        last = '' if len(rslt)==0 else rslt[-1]
        next_ele = modify_strlen_ths(last,words[pnt],ths)
        rslt = rslt[:-1] + next_ele
        pnt += 1
    return '\n'.join(rslt)[1:]

def choose_plot_grid(n:int,r_max=8,c_max=17,res_ths=2):
    #ver2
    r_min = np.ceil(n/c_max)
    sppt = np.arange(r_min,r_max+1) #need error process
    col_nums = np.ceil(n/sppt)
    res = col_nums * sppt -n
    min_idx = np.where((res==np.min(res)) | (res <= res_ths))[0]
    row_cand, col_cand = sppt[min_idx], col_nums[min_idx]
    if len(min_idx) > 1 :
        res = np.abs(row_cand-col_cand)
        i = np.where(res==np.min(res))[0][0]
    else : i = 0
    return int(row_cand[i]), int(col_cand[i])

def pair_plot_feat_hue(fig,axes,data:dict,pair_plot,axis_share=False,hue_label_dict=None, **kwargs):
    #ver2
    if (fig is None) or (axes is None) :
        if len(data) == 1: fig,axes = plt.subplots(figsize=(4,4))
        else : 
            num_r, num_c = choose_plot_grid(len(data))
            fig, axes = plt.subplots(num_r,num_c,figsize=(4*num_c,4*num_r),sharex=axis_share,sharey=axis_share)
    for n,key in enumerate(data.keys()):
        ax = axes.flatten()[n] if len(data) > 1 else axes        
        plt.setp(ax.get_xticklabels(),ha = 'left',rotation = 90)
        if n >= len(data) : continue
        pair_plot(x=data[key][0], y = data[key][1],ax =ax, **kwargs)
        feat_name = str(key) 
        if hue_label_dict: color = 'b' if hue_label_dict[feat_name] else 'k'
        else : color = 'k'
        ax.set_xlabel(str_cutter(feat_name,20),loc='left',fontsize = 8.3,color=color)
    return fig,axes

def plot_area(val,decrease=True,start=0,end=-1,log=True):
  mov_val, lim_val = get_vals_range(val,decrease)
  fig,axes = plt.subplots(1,4,figsize=(16,4))
  mov_df=pd.DataFrame(mov_val)[start:end]
  mov_amp = pd.DataFrame(get_amp(mov_df))
  lim_df=pd.DataFrame(lim_val)[start:end]
  lim_amp = pd.DataFrame(get_amp(lim_df))
  axes[0].plot(mov_df,linewidth=0.45,label=mov_df.columns)
  axes[2].plot(mov_amp,linewidth=0.45,label=mov_amp.columns)
  axes[1].plot(lim_df,linewidth=0.45,label=lim_df.columns)
  axes[3].plot(lim_amp,linewidth=0.45,label=lim_amp.columns)
  
  titles = list(map(lambda x : f'{x[1]}_{x[0]}',
                product(['value','width'],['moving','lim'])))
  for title,ax in zip(titles,axes):
    ax.set_title(title,fontsize=17)
    if log : ax.set_yscale('log')
    for line in ax.get_lines():
      name = line.get_label()
      if 'avg' in name : line.set(lw=0.65,color='#55A235')
      elif 'real' in name : line.set(lw = 0.3, alpha=0.8)
    ax.legend(fontsize=10)
  return fig,axes