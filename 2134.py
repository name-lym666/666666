import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 将图的title中文显示
plt.rcParams['axes.unicode_minus'] = False
df1 = pd.read_excel('result.xlsx', usecols=['FrameID', 'Y(m)', 'X(m)', 'V(m/s)', 'TTC(s)'])  # 导入表格
print('测量值result(df1)为：')
print(df1)
result2 = pd.DataFrame(df1, columns=['FrameID', 'Y(m)', 'X(m)', 'V(m/s)', 'TTC(s)'])
result2.to_excel('result2.xlsx')
df2 = pd.read_csv('accurate.csv', usecols=['Frame ID', 'Y(m)', 'X(m)', 'Vy(m/s)', 'TTCy(s)'])
df2[['Vy(m/s)', 'TTCy(s)']] = df2[['TTCy(s)', 'Vy(m/s)']]  # 交换两列
resultok = df1.values  # 数组存储,resultok为result数组数据结构，accurateok为accurate数组数据结构
accurateok = df2.values
accback = resultok - resultok  # 建立空数组存储帧同步值,accback用于存accurate与result帧同步后的值
for i in range(0, 962):  # 帧同步
    for j in range(0, 1054):
        if resultok[i][0] == accurateok[j][0]:
            for k in range(0, 5):
                accback[i][k] = accurateok[j][k]
accurate2 = pd.DataFrame(accback, columns=['Frame ID', 'Y(m)', 'X(m)', 'Vy(m/s)', 'TTCy(s)'])  # 真值
print('帧同步后的值：accurate2')
print(accurate2)
accurate2.to_excel('accurate2.xlsx', index=0)
# **********************************************************************************************************************************
fs = abs(df1.values - accurate2.values)  # four series
for i in range(0, 962):  # 重新赋  FrameID   值
    fs[i][0] = resultok[i][0]  # VVV中的数组形式
e = pd.DataFrame(fs, columns=['FrameID', 'Y差值', 'X差值', 'V/Vy', 'TTC/TTCy'])  # 做差后的变量
print('四个作差后的值为e:')
print(e)
# **********************************************************************************************************************************
e.to_excel('VVV.xlsx', index=0)
lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 建立一个列表存放极值
i = 0
for cols in e.columns:  # 将最大值最小值按顺序存放列表中
    dd = e[cols]
    lst[i] = dd.max()
    i = i + 1
    lst[i] = dd.min()
    i = i + 1
las = {'Y差值': [lst[2], lst[3]], 'X差值': [lst[4], lst[5]], 'V/Vy': [lst[6], lst[7]],
       'TTC/TTCy': [lst[8], lst[9]]}  # 生成字典,输出excel
qq = pd.DataFrame(las, index=['max', 'min'])
qq.to_excel('las.xlsx')  # 导出excel表
print('极值为qq：')
print(qq)
trueY = accurate2.values[:, 1]  # Y  真值
fs1 = fs[:, 1]  # Y 差值
fs2 = fs[:, 2]  # X 差值
fs3 = fs[:, 3]  # V/Vy
fs4 = fs[:, 4]  # TTC/TTCy 差值
fs_All = [fs1, fs2, fs3, fs4]
# true_All=[trueY,trueX,trueV,trueT]
for i in range(0, 962):
    if trueY[i] < 120:
        limit = i  # 界限值
        break
numlimit = 962 - limit  # numlimit为小于Y真值120m的个数
print('120m处值：', limit, '小于120m的数量：', numlimit)
xY = [0] * 559  # 横坐标 Y
xX = [0] * 559
xV = [0] * 559
xT = [0] * 559
yy_Y = [0] * 559  # y 差值
yx_X = [0] * 559  # x 差值
yv_V = [0] * 559  # V 差值
yt_T = [0] * 559  # TTC 差值
y_per = [0] * 559
x_per = [0] * 559
v_per = [0] * 559
t_per = [0] * 559
j = 0
for i in range(limit, 962):
    xY[j] = accurate2.values[i][1]
    xX[j] = accurate2.values[i][2]
    xV[j] = accurate2.values[i][3]
    xT[j] = accurate2.values[i][4]
    yy_Y[j] = fs1[i]
    yx_X[j] = fs2[i]
    yv_V[j] = fs3[i]
    yt_T[j] = fs4[i]
    y_per[j] = fs1[i] / xY[j]
    x_per[j] = fs2[i] / xX[j]
    v_per[j] = fs3[i] / xV[j]
    t_per[j] = fs4[i] / xT[j]

    # y_per[j] = fs1[i] / abs(xY[j])
    # x_per[j] = fs2[i] / abs(xX[j])
    # v_per[j] = fs3[i] / abs(xV[j])
    # t_per[j] = fs4[i] / abs(xT[j])
    j = j + 1
y_all = [yy_Y, yx_X, yv_V, yt_T]
y_all = np.array(y_all)
# print('Y差值y_all')
# print(y_all)
y_all = pd.DataFrame(y_all)
y_x_all = [xY, xX, xV, xT]
y_x_all = np.array(y_x_all)
frame_id = range(12358, 12917)
lac = {'FrameID': frame_id, 'Y坐标': xY, 'Y差值': yy_Y, 'Y百分比': y_per, 'X坐标': xX, 'X差值': yx_X, 'X百分比': x_per, 'Vy': xV,
       'V/Vy差值': yv_V,
       'V百分比': v_per, 'TTCy': xT, 'TTC差值': yt_T, 'T百分比': t_per, }
qp = pd.DataFrame(lac)
qp.to_excel('SSS.xlsx')
Y1 = ['Y坐标', 'X坐标', 'Vy', 'TTCy']
X1 = ['Y差值', 'X差值', 'V/Vy差值', 'TTC差值']
X2 = ['Y百分比', 'X百分比', 'V百分比', 'T百分比']
combine1 = e
combine1.insert(1, 'Y真值', accurate2['Y(m)'].values)  # 插入Y真值操作
combine1.to_excel('VVY.xlsx', index=0)  # 导出Excel表方便查看
limitvalue = [120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
numcombine = [0] * 12  # 存储每10m的真值
for i in range(0, 12):
    for j in range(limit, 962):
        if (combine1['Y真值'][j] <= limitvalue[i] and combine1.values[j][1] >= limitvalue[i + 1]):
            numcombine[i] = numcombine[i] + 1  # numcombine 统计各段帧数
print('每段帧的个数：numcombine', numcombine)
t = np.array([0] * 12)
for i in range(0, 12):
    t[i] = int(0.95 * numcombine[i]) - 1
numcombine2 = [1] * 12
numcombine2 = numcombine
for i in range(1, 12):  # 整理numcombine1
    numcombine2[i] = numcombine[i] + numcombine[i - 1]
# print(numcombine2)
numcombine1 = [1] * 13
for i in range(1, 13):
    numcombine1[i] = numcombine2[i - 1]
numcombine1 = list(
    np.array(numcombine1) - np.array([1] * 13) + np.array([limit] * 13))  # numcombine1  为遍历对象的   " range "
numcombine1 = np.array(numcombine1) + np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # 遍历区间
print('遍历start-end numcombine1:', numcombine1)
name = ['Y差值', 'X差值', 'V/Vy', 'TTC/TTCy']
print('**********************显示DataFrame类型数据组合后的combine1: ' + '**********************')
print(combine1)
name2 = ['FrameID', 'Y真值', 'Y差值', 'X差值', 'V/Vy', 'TTC/TTCy']
big_list = [[0.0] * 6] * numlimit
big_list = np.array(big_list)
# print(big_list)
lst1 = np.array(
    [np.array([0.0] * 40), np.array([0.0] * 39), np.array([0.0] * 39), np.array([0.0] * 38), np.array([0.0] * 39),
     np.array([0.0] * 37), np.array([0.0] * 36), np.array([0.0] * 36), np.array([0.0] * 35), np.array([0.0] * 36),
     np.array([0.0] * 42),
     np.array([0.0] * 142)])
for j in range(0, 559):
    for k in range(0, 6):
        big_list[j][k] = combine1[name2[k]][j + limit]
onehund = pd.DataFrame(big_list, columns=name2)
onrhundred = onehund.to_excel('onh.xlsx')
print('**********************显示DataFrame类型数据组合后(小于120M)的:onehund:')
print(onehund)
print(t)
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
for i in range(0, 12):
    k = 0
    for j in range(numcombine1[i], numcombine1[i + 1]):
        lst1[i][k] = combine1['Y差值'][j]
        k = k + 1
for i in range(0, 12):
    lst1[i].sort()
# print(lst1)
nf_values = np.array([[0.0] * 4] * 12)
i = 0
for j in t:
    nf_values[i][0] = lst1[i][j]
    i = i + 1
for i in range(0, 12):
    k = 0
    for j in range(numcombine1[i], numcombine1[i + 1]):
        lst1[i][k] = combine1['X差值'][j]
        k = k + 1
for i in range(0, 12):
    lst1[i].sort()
i = 0
for j in t:
    nf_values[i][1] = lst1[i][j]
    i = i + 1
for i in range(0, 12):
    k = 0
    for j in range(numcombine1[i], numcombine1[i + 1]):
        lst1[i][k] = combine1['V/Vy'][j]
        k = k + 1
for i in range(0, 12):
    lst1[i].sort()
i = 0
for j in t:
    nf_values[i][2] = lst1[i][j]
    i = i + 1
for i in range(0, 12):
    k = 0
    for j in range(numcombine1[i], numcombine1[i + 1]):
        lst1[i][k] = combine1['TTC/TTCy'][j]
        k = k + 1
for i in range(0, 12):
    lst1[i].sort()
i = 0
for j in t:
    nf_values[i][3] = lst1[i][j]
    i = i + 1
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
y_max = ['120-110m', '110-100m', '100-90m', '90-80m', '80-70m', '70-60m', '60-50m', '50-40m', '40-30m', '30-20m',
         '20-10m', '10-0m']
dftemp2 = pd.DataFrame(nf_values, columns=['Y差值', 'X差值', 'V/Vy', 'TTC/TTCy'], index=y_max)
dftemp2.to_excel('lower.xlsx')
# **********************************************************************************************************************************
print('显示name', name)
print('**********************显示最大值百分之95的值：dftemp2' + '  ********************')
print(dftemp2)
frdownnfnum = [[0] * 4] * 12
frdownnfnum = np.array(frdownnfnum)
# ***********************************************************************************************************************************
# 统计小于百分之95帧的数量（没用）
for k in range(0, 4):
    for i in range(0, 12):
        for j in range(numcombine1[i], numcombine1[i + 1]):
            if (combine1[name[k]][j] <= dftemp2[name[k]][i]):
                frdownnfnum[i][k] = frdownnfnum[i][k] + 1

dffrnum = pd.DataFrame(frdownnfnum, columns=['Y差值(numframe)', 'X差值(numframe)', 'V/Vy(numframe)', 'TTC/TTCy(numframe)'],
                       index=y_max)
# **********************************************************************************************************************************
# print('***********************显示帧的数量（差值在百分之95内）：dffrnum'+'  ************************')
# print(dffrnum)
# **********************************************************************************************************************************
print('******************************求出Y差值,X差值,V/Vy,TTC/TTCy连续大于95%的帧数**********************************')
dis_subY_all_frame = sum(dffrnum['Y差值(numframe)'])
subY_all_frame = 559 - dis_subY_all_frame
dis_subX_all_frame = sum(dffrnum['X差值(numframe)'])
subX_all_frame = 559 - dis_subX_all_frame
dis_subV_all_frame = sum(dffrnum['V/Vy(numframe)'])
subV_all_frame = 559 - dis_subV_all_frame
dis_subT_all_frame = sum(dffrnum['TTC/TTCy(numframe)'])
subT_all_frame = 559 - dis_subT_all_frame
All_sub = [subY_all_frame, subX_all_frame, subV_all_frame, subT_all_frame]
print('统计Y差值大于百分之95帧的个数，用于建立列表:', subY_all_frame)
print('统计X差值大于百分之95帧的个数，用于建立列表:', subX_all_frame)
print('统计V/Vy大于百分之95帧的个数，用于建立列表:', subV_all_frame)
print('统计TTC/TTCy大于百分之95帧的个数，用于建立列表:', subT_all_frame)
Y_frameID = [0] * subY_all_frame
X_frameID = [0] * subX_all_frame
V_frameID = [0] * subV_all_frame
T_frameID = [0] * subT_all_frame
All_frameID = [Y_frameID, X_frameID, V_frameID, T_frameID]
name1 = ['Y差值', 'X差值', 'V/Vy', 'TTC/TTCy']
for q in range(0, 4):
    k = 0
    for i in range(0, 12):
        for j in range(numcombine1[i], numcombine1[i + 1]):
            if (combine1[name1[q]][j] > dftemp2[name1[q]][i]):
                All_frameID[q][k] = combine1['FrameID'][j]
                k = k + 1
print('Y差值大于95%的帧数 :', Y_frameID)
print('X差值大于95%的帧数 :', X_frameID)
print('V/Vy大于95%的帧数 :', V_frameID)
print('TTC/TTCy大于95%帧数：', T_frameID)
Y_contue = [0] * subY_all_frame
X_contue = [0] * subX_all_frame
V_contue = [0] * subV_all_frame
T_contue = [0] * subT_all_frame
All_contue = [Y_contue, X_contue, V_contue, T_contue]
for q in range(0, 4):
    for i in range(1, All_sub[q]):
        if (All_frameID[q][i] - All_frameID[q][i - 1] == 1):
            for j in range(0, All_sub[q]):
                if (All_frameID[q][i - 1] != All_contue[q][j]):
                    All_contue[q][i - 1] = All_frameID[q][i - 1]
                if (All_frameID[q][i] != All_contue[q][j]):
                    All_contue[q][i] = All_frameID[q][i]
All_num = [0, 0, 0, 0]
for q in range(0, 4):
    for i in range(0, 3):
        for lst in All_contue[q]:
            if lst == 0:
                All_contue[q].remove(lst)
    print('连续帧数为:', All_contue[q])
    All_num[q] = len(All_contue[q])
    print(len(All_contue[q]))
print('******************************Y差值,X差值,V/Vy,TTC/TTCy连续大于95%的帧数求解完毕**********************************')
print('**Y差值,X差值,V/Vy,TTC/TTCy帧数值:All_num', All_num)
# frame_temp=[0]*22
# for i in range(0,All_num[0]):
#     for k in [1,2,3]:
#         for j in range(0,All_num[k]):
#             if All_contue[0][i]==All_contue[k][j]:
#                 frame_temp[i]=All_contue[0][i]
# for i in range(0, 3):  # 去零
#     for lst in frame_temp:
#         if lst == 0:
#             frame_temp.remove(lst)
# print('****所有数据都满足连续大于95%帧值为：',frame_temp)
# print('****连续帧数量为:',len(frame_temp))
# k=0
# fig, ax1 = plt.subplots(2, 2)
# for i in range(0,2):
#     for j in range(0,2):
#         ax2=ax1
#         colors='red'
#         ax1[i][j].set_xlabel(Y1[k])
#         ax1[i][j].set_ylabel(X1[k],color=colors) # 纵坐标参数（title）变色
#         ax1[i][j].plot(qp[Y1[k]],qp[X1[k]],color=colors) # 曲线变色
#         ax1[i][j].tick_params(axis='y',labelcolor=colors) # 纵坐标轴变色
#         ax2[i][j]=ax1[i][j].twinx()
#         colors='blue'
#         if k == 1:
#             ax2[i][j].axis([-0.2, 1, -2, 2])
#         elif k==2:
#             ax2[i][j].axis([-10, 4, -2, 2])
#         ax2[i][j].set_ylabel(X2[k],color=colors)
#         ax2[i][j].plot(qp[Y1[k]],qp[X2[k]],color=colors)
#         ax2[i][j].tick_params(axis='y',labelcolor=colors) # 纵坐标轴变色
#         k=k+1
# plt.show()
llst = np.array([[12378, 12901], [12359, 12881], [12358, 12902], [12396, 12866]])
llst2 = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
for m in range(0, 4):
    for n in range(0, 2):
        for i in range(0, 559):
            if onehund['FrameID'][i] == llst[m][n]:
                llst2[m, n] = onehund['Y真值'][i]
print(llst2)
str_name = ['Y差值连续帧数距离范围：', 'X差值连续帧数距离范围：', 'V差值连续帧数距离范围：', 'TTC差值连续帧数距离范围：']
# print('Y真值的范围为：')
# for i in range(0, 4):
#     print(str_name[i])
#     for j in range(0, 2):
#         print(llst2[i][j])
# 精度统计
name3 = ['FrameID', 'Y百分比', 'X百分比', 'V百分比', 'T百分比']
big_list2 = [[0.0] * 5] * numlimit
big_list2 = np.array(big_list2)
for j in range(0, 559):
    for k in range(0, 5):
        big_list2[j][k] = qp[name3[k]][j]
jd = pd.DataFrame(big_list2, columns=name3)
jd.to_excel('jingdu.xlsx')
jdrange = [jd[name3[0]][0], jd[name3[0]][0] + 59, 12916]  # 遍历范围
max_all = np.array([max(abs(jd[name3[1]])), max(abs(jd[name3[2]])), max(abs(jd[name3[3]])), max(abs(jd[name3[4]]))])
each_frame=np.array([[0.0]*5]*559)
#归一化
for i in range(0,4):
    for j in range(0,559):
        each_frame[j][i]=abs(jd[name3[i+1]][j])/max_all[i]
for i in range(0,559):
    each_frame[i][4]=(each_frame[i][0]+each_frame[i][1]+each_frame[i][2]+each_frame[i][3])/4
ef=pd.DataFrame(each_frame,columns=['Y误差','X误差','V误差','TTC误差','误差均值'])
print('各误差为：')
print(ef)
print('前60帧的误差水平：')
sum_60 = np.array([0.0, 0.0, 0.0, 0.0])
max_60 = np.array([max(jd[name3[1]]), max(jd[name3[2]]), max(abs(jd[name3[3]])), max(jd[name3[4]])])
for i in range(0, 4):
    k = 0
    for f in jd[name3[0]]:
        if f <= jdrange[1]:
            sum_60[i] = jd[name3[i + 1]][k] + sum_60[i]
            k = k + 1
sum_60_acc = np.array([0.0] * 4)
sum_60_acc = abs(sum_60) / max_60 / 60
print(name3[1], '               ', name3[2], '                 ', name3[3], '             ', name3[4],'             ','平均误差')
print(sum_60_acc[0], ' ', sum_60_acc[1], ' ', sum_60_acc[2], ' ', sum_60_acc[3],'',sum(sum_60_acc)/4)
print('前60帧的误差水平统计完毕。')
print('后面帧的误差水平：')
sum_sall=np.array([0.0, 0.0, 0.0, 0.0])
for i in range(0, 4):
    k = 0
    for f in jd[name3[0]]:
        if f > jdrange[1] and f <= jdrange[2]:
            sum_sall[i] = jd[name3[i + 1]][k] + sum_sall[i]
            k = k + 1
t=jdrange[2]-jdrange[1]+1
sum_sall_acc = np.array([0.0] * 4)
sum_sall_acc = abs(sum_sall) / max_60 /t
print(name3[1], '               ', name3[2], '               ', name3[3], '               ', name3[4],'             ','平均误差')
print(sum_sall_acc[0], ' ', sum_sall_acc[1], ' ', sum_sall_acc[2], ' ', sum_sall_acc[3],' ',sum(sum_sall_acc)/4)
print('后面帧的误差水平统计完毕。')


