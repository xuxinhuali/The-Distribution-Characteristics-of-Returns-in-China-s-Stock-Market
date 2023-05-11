"""
This module aims to study the distribution
characteristics of Chinese stock market returns
Author: Xuxin
Date: 2023-03-15
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from scipy.stats import norm, kstest, powerlaw

plt.rcParams.update({"font.family": "STIXGeneral",
                     "font.size": 20,
                     "mathtext.fontset": "cm"})
# 上证日度数据
ssec_daily = pd.read_csv(r'1LShowData_SSEC_daily.csv',
                         encoding='GB2312', usecols=[6])

# 上证指数日度
p_daily = ssec_daily['收盘'].values  # 日度数据数组
print(p_daily)
# 日度收益率图片
r_daily = np.log(p_daily[1:]) - np.log(p_daily[:-1])

# mat数据
matdata = loadmat(r'SSEC_min.mat')
p_min = matdata['p']

# 1分钟数据
p_min = p_min[5:, 0]  # 去除前五个不正常数据
p_min = p_min[(p_min >= 85) & (p_min <= 8000)]
print(p_min)


# 计算收益率的函数
def calculate_r_min(p_min, bin):
    r_min_bin = np.log(p_min[bin::bin]) - np.log(p_min[:-bin:bin])
    return r_min_bin[(r_min_bin >= -0.1) & (r_min_bin <= 0.1)]


# （1分钟、5分钟、10分钟、30分钟、60分钟、120分钟、240分钟）
r_min_1 = calculate_r_min(p_min=p_min, bin=1)
r_min_5 = calculate_r_min(p_min=p_min, bin=5)
r_min_10 = calculate_r_min(p_min=p_min, bin=10)
r_min_30 = calculate_r_min(p_min=p_min, bin=30)
r_min_60 = calculate_r_min(p_min=p_min, bin=60)
r_min_120 = calculate_r_min(p_min=p_min, bin=120)
r_min_240 = calculate_r_min(p_min=p_min, bin=240)

# 将收益率数据存储在字典中,计算统计值
returns_dict = {'1min': r_min_1, '5min': r_min_5,
                '10min': r_min_10, '30min': r_min_30,
                '60min': r_min_60, '120min': r_min_120,
                '240min': r_min_240, 'daily': r_daily}
count_returns = {}
mean_returns = {}
std_returns = {}
max_returns = {}
min_returns = {}
for key, value in returns_dict.items():
    count_returns[key] = len(value)
    mean_returns[key] = np.mean(value)
    std_returns[key] = np.std(value)
    max_returns[key] = np.max(value)
    min_returns[key] = np.min(value)
# 将统计结果存储到一个数据框中，然后导出表格
Statistic_result = pd.DataFrame({'样本个数': count_returns, '平均收益率': mean_returns,
                                 '标准差': std_returns, '最大收益率': max_returns, '最小收益率': min_returns})
Statistic_result.to_csv('Statistic_result.csv', encoding='gbk')


# num_bin是要多少个间隔
def myfun_emp_pdf(data_sample, num_bin=31):
    bin = np.linspace(np.min(data_sample), np.max(data_sample), num_bin)  # 等间隔
    x_emp = np.zeros(len(bin) - 1)
    y_emp = np.zeros(len(bin) - 1)
    for i in range(len(bin) - 1):
        x_emp[i] = (bin[i] + bin[i + 1]) / 2
        y_emp[i] = (np.sum((data_sample >= bin[i]) & (data_sample < bin[i + 1]))) / len(data_sample) / (
                    bin[i + 1] - bin[i])
    return x_emp, y_emp


# 分割
x_emp_daily, y_emp_daily = myfun_emp_pdf(r_daily)
x_emp_min_1, y_emp_min_1 = myfun_emp_pdf(r_min_1, num_bin=31)
x_emp_min_5, y_emp_min_5 = myfun_emp_pdf(r_min_5, num_bin=31)
x_emp_min_10, y_emp_min_10 = myfun_emp_pdf(r_min_10, num_bin=31)
x_emp_min_30, y_emp_min_30 = myfun_emp_pdf(r_min_30, num_bin=31)
x_emp_min_60, y_emp_min_60 = myfun_emp_pdf(r_min_60, num_bin=31)
x_emp_min_120, y_emp_min_120 = myfun_emp_pdf(r_min_120, num_bin=31)
x_emp_min_240, y_emp_min_240 = myfun_emp_pdf(r_min_240, num_bin=31)

ind1 = y_emp_min_1 > 0
ind5 = y_emp_min_5 > 0
ind10 = y_emp_min_10 > 0
ind30 = y_emp_min_30 > 0
ind60 = y_emp_min_60 > 0
ind120 = y_emp_min_120 > 0
ind240 = y_emp_min_240 > 0

x_emp_min_1, y_emp_min_1 = x_emp_min_1[ind1], y_emp_min_1[ind1]
x_emp_min_5, y_emp_min_5 = x_emp_min_5[ind5], y_emp_min_5[ind5]
x_emp_min_10, y_emp_min_10 = x_emp_min_10[ind10], y_emp_min_10[ind10]
x_emp_min_30, y_emp_min_30 = x_emp_min_30[ind30], y_emp_min_30[ind30]
x_emp_min_60, y_emp_min_60 = x_emp_min_60[ind60], y_emp_min_60[ind60]
x_emp_min_120, y_emp_min_120 = x_emp_min_120[ind120], y_emp_min_120[ind120]
x_emp_min_240, y_emp_min_240 = x_emp_min_240[ind240], y_emp_min_240[ind240]

# 创建子图，将函数图画在同一张图上
plt.semilogy(x_emp_min_1, y_emp_min_1, marker='s', c='#FFA500', label='1min')
plt.semilogy(x_emp_min_5, y_emp_min_5, marker='d', c='#FFC0CB', label='5min')
plt.semilogy(x_emp_min_10, y_emp_min_10, marker='h', c='#000000', label='10min')
plt.semilogy(x_emp_min_30, y_emp_min_30, marker='+', c='#008000', label='30min')
plt.semilogy(x_emp_min_60, y_emp_min_60, marker='^', c='#800080', label='60min')
plt.semilogy(x_emp_min_120, y_emp_min_120, marker='*', c='#000080', label='120min')
plt.semilogy(x_emp_min_240, y_emp_min_240, marker='p', c='#FF0000', label='240min')
plt.semilogy(x_emp_daily, y_emp_daily, marker='o', c='#F0E68C', label='daily')
plt.xlabel(r'$r$', fontsize=20)
plt.ylabel(r'$p(r)$', fontsize=20)
plt.legend(loc='upper right', fontsize=10)
plt.savefig(f'Fig_PDF_Return_CNMarkets.jpg', dpi=600, bbox_inches='tight')
plt.show()

# 拟合
mu_daily, sigma_daily = norm.fit(r_daily)
mu_min_1, sigma_min_1 = norm.fit(r_min_1)
mu_min_5, sigma_min_5 = norm.fit(r_min_5)
mu_min_10, sigma_min_10 = norm.fit(r_min_10)
mu_min_30, sigma_min_30 = norm.fit(r_min_30)
mu_min_60, sigma_min_60 = norm.fit(r_min_60)
mu_min_120, sigma_min_120 = norm.fit(r_min_120)
mu_min_240, sigma_min_240 = norm.fit(r_min_240)

x_fit_daily = np.linspace(-0.1, 0.1, 300)
x_fit_min_1 = np.linspace(-0.1, 0.1, 300)
x_fit_min_5 = np.linspace(-0.1, 0.1, 300)
x_fit_min_10 = np.linspace(-0.1, 0.1, 300)
x_fit_min_30 = np.linspace(-0.1, 0.1, 300)
x_fit_min_60 = np.linspace(-0.1, 0.1, 300)
x_fit_min_120 = np.linspace(-0.1, 0.1, 300)
x_fit_min_240 = np.linspace(-0.1, 0.1, 300)

y_fit_daily = norm.pdf(x_fit_daily, loc=mu_daily, scale=sigma_daily)
y_fit_min_1 = norm.pdf(x_fit_min_1, loc=mu_min_1, scale=sigma_min_1)
y_fit_min_5 = norm.pdf(x_fit_min_5, loc=mu_min_5, scale=sigma_min_5)
y_fit_min_10 = norm.pdf(x_fit_min_10, loc=mu_min_10, scale=sigma_min_10)
y_fit_min_30 = norm.pdf(x_fit_min_30, loc=mu_min_30, scale=sigma_min_30)
y_fit_min_60 = norm.pdf(x_fit_min_60, loc=mu_min_60, scale=sigma_min_60)
y_fit_min_120 = norm.pdf(x_fit_min_120, loc=mu_min_120, scale=sigma_min_120)
y_fit_min_240 = norm.pdf(x_fit_min_240, loc=mu_min_240, scale=sigma_min_240)


# 正态分布
def nihe(text_fit_x, text_emp_y, r_min, x_emp, y_emp, x_fit, y_fit, name):
    # 使用KS检验
    test_statistic, p_value = kstest(r_min, 'norm')
    # 显示结果
    alpha = 0.05
    if p_value > alpha:
        plt.text(-0.090, 60, 'p = {:.2f}\nfollows a normal\ndistribution'.format(p_value), fontsize=15)
    else:
        plt.text(-0.090, 60, 'p = {:.2f}\ndoes not follow a\nnormal distribution'.format(p_value), fontsize=15)
    plt.semilogy(x_emp, y_emp, 'o-r', lw=2, ms=7, mfc='k', label=r'$r_d$ Emp PDF')
    plt.semilogy(x_fit, y_fit, '--m', lw=2, label=r'$r_d$ Norm Fits')
    plt.xlim([-0.1, 0.1])
    plt.xticks([-0.1, -0.05, 0, 0.05, .1])
    plt.ylim([10. ** -4, 10 ** 3])
    plt.yticks(10. ** np.arange(-4, 4, 2))
    plt.xlabel(r'$r$', fontsize=20)
    plt.ylabel(r'$p(r)$', fontsize=20)
    plt.text(-0.080, text_emp_y, name, fontsize=20)
    plt.text(text_fit_x, 0.00025, 'fit', fontsize=20)
    plt.legend(loc='upper right', fontsize=10)
    plt.savefig(f'{name}_Fig_PDF_Return_CNMarkets.jpg', dpi=600, bbox_inches='tight')
    plt.show()


# 八个拟合
nihe(-0.020, 0.050, r_min_1, x_emp_min_1, y_emp_min_1, x_fit_min_1, y_fit_min_1, name='1min')
nihe(-0.035, 0.053, r_min_5, x_emp_min_5, y_emp_min_5, x_fit_min_5, y_fit_min_5, name='5min')
nihe(-0.040, 0.055, r_min_10, x_emp_min_10, y_emp_min_10, x_fit_min_10, y_fit_min_10, name='10min')
nihe(-0.045, 0.055, r_min_30, x_emp_min_30, y_emp_min_30, x_fit_min_30, y_fit_min_30, name='30min')
nihe(-0.060, 0.150, r_min_60, x_emp_min_60, y_emp_min_60, x_fit_min_60, y_fit_min_60, name='60min')
nihe(-0.050, 0.500, r_min_120, x_emp_min_120, y_emp_min_120, x_fit_min_120, y_fit_min_120, name='120min')
nihe(-0.065, 1.000, r_min_240, x_emp_min_240, y_emp_min_240, x_fit_min_240, y_fit_min_240, name='240min')
nihe(-0.070, 2.000, r_daily, x_emp_daily, y_emp_daily, x_fit_daily, y_fit_daily, name='daily')


def fit_line(x, y):  # 线性拟合
    n = len(x)
    sumx = np.sum(x)
    sumx_2 = np.sum(x ** 2)
    sumy = np.sum(y)
    sumx_y = np.sum(x * y)
    a = np.array([[n, sumx], [sumx, sumx_2]])
    b = np.array([[sumy], [sumx_y]])
    parLSQ = np.dot(np.linalg.inv(a), b)
    return parLSQ[0], parLSQ[1]


def draw_loglog(x_emp, y_emp, name):  # 输出一张拟合图和散点图
    x_pos, y_pos = x_emp[x_emp > 0], y_emp[x_emp > 0]  # 正尾
    x_neg, y_neg = [np.abs(x) for x in x_emp[x_emp < 0]], y_emp[x_emp < 0]  # 负尾
    postail_b, postail_k = fit_line(np.log(x_pos), np.log(y_pos))  # 正尾截距和斜率
    negtail_b, negtail_k = fit_line(np.log(x_neg), np.log(y_neg))  # 负尾
    y_pos_fit = np.exp(postail_b + postail_k * np.log(np.array(x_pos)))
    y_neg_fit = np.exp(negtail_b + negtail_k * np.log(np.array(x_neg)))
    plt.loglog(x_pos, y_pos, 'or', lw=2, ms=7, mfc='k')
    plt.loglog(x_pos, y_pos_fit, '--r', lw=2, ms=7, mfc='k', label=r'positive')
    plt.loglog(x_neg, y_neg, 'ob', lw=2, ms=7, mfc='k')
    plt.loglog(x_neg, y_neg_fit, '--b', lw=2, ms=7, mfc='k', label=r'negative')
    plt.xlim([10. ** -3, 10 ** -1])
    plt.xticks(10. ** np.arange(-3, -1, 1))
    plt.ylim([10. ** -3, 10 ** 3])
    plt.yticks(10. ** np.arange(-3, 3, 1))
    plt.text(0.0040, 0.0250, name, fontsize=20)
    plt.text(0.0100, 80.000, 'pos_para:' + str(round((float(postail_k[0])), 2)), fontsize=15)
    plt.text(0.0100, 40.000, 'neg_para:' + str(round((float(negtail_k[0])), 2)), fontsize=15)
    plt.xlabel(r'$r$', fontsize=20)
    plt.ylabel(r'$p(r)$', fontsize=20)
    plt.legend(loc='upper right', fontsize=10)
    plt.savefig(f'{name}_Fig_PDF_Return_TailFeature_CNMarkets.jpg', dpi=600, bbox_inches='tight')
    plt.show()


draw_loglog(x_emp_min_1, y_emp_min_1, '1min')
draw_loglog(x_emp_min_5, y_emp_min_5, '5min')
draw_loglog(x_emp_min_10, y_emp_min_10, '10min')
draw_loglog(x_emp_min_30, y_emp_min_30, '30min')
draw_loglog(x_emp_min_60, y_emp_min_60, '60min')
draw_loglog(x_emp_min_120, y_emp_min_120, '120min')
draw_loglog(x_emp_min_240, y_emp_min_240, '240min')
draw_loglog(x_emp_daily, y_emp_daily, 'daily')
