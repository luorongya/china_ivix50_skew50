# -*- coding: utf-8 -*-

import pandas as pd
# import numpy as np
# import datetime
# import os
from math import exp, sqrt, log
from WindPy import *
# import matplotlib.pyplot as plt


def get_time():
    # 获取当前时间
    return time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime(time.time()))


def check_connection():
    if not w.isconnected():
        w.start()
    return


def close_connection():
    if w.isconnected():
        w.stop()
    return


def fetch_raw_contracts(key, sym, end_date):
    check_connection()
    # key = u'沪50ETF'
    # sym = '510050.SH'
    req_start = str(int(end_date[0:4]) - 1) + end_date[4:]
    req_dates = 'startdate=' + req_start + ';enddate='+ end_date + ';'
    req_fields = 'field=wind_code,trade_code,call_or_put,exercise_price,expire_date,contract_state'
    params = req_dates + 'exchange=sse;windcode=' + sym + ';status=trading;' + req_fields
    fetch = w.wset('optioncontractbasicinfo', params)
    basic = pd.DataFrame()
    basic['option_code'] = fetch.Data[0]  # 期权代码
    basic['trade_code'] = fetch.Data[1]  # 交易代码
    basic['call_or_put'] = fetch.Data[2]  # 认沽认购
    basic['strike_price'] = fetch.Data[3]  # 行权价
    basic['expire_date'] = fetch.Data[4]  # 到期日期
    basic['contract_state'] = fetch.Data[5]  # 合约状态
    # basic['data_source'] = 'Wind'
    # basic['created_date'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(get_time(), key, len(basic), 'contracts to ' +
          basic['expire_date'][len(basic) - 1].strftime('%Y-%m-%d') + '.')
    return basic


def get_contract(options, end_date):
    print(get_time(), u'期权合约数据开始下载。')
    basic = []
    for key in options.keys():
        sym = options[key]
        print(get_time(), 'Download ' + key + ' starts.')
        basic.append(fetch_raw_contracts(key, sym, end_date))
    print(get_time(), u'期权合约数据获取完毕。')
    basic[0].to_excel('E:\\Projects\\volatility\\data\\sh50_contracts.xlsx')
    basic[1].to_excel('E:\\Projects\\volatility\\data\\sh300_contracts.xlsx')
    return basic


def fetch_raw_quote(key, sym, start_date, end_date):
    check_connection()
    params = 'startdate=' + start_date + ';enddate=' + end_date + ';exchange=sse;windcode=' + sym + \
             ';field=date,option_code,tradecode,exerciseprice,settlement_price'
    fetch = w.wset("optiondailyquotationstastics", params)
    quote = pd.DataFrame()
    quote['trade_date'] = fetch.Data[0]  # 交易日期
    quote['option_code'] = fetch.Data[1]  # 期权代码
    quote['trade_code'] = fetch.Data[2]  # 期权简称
    quote['strike'] = fetch.Data[3]  # 期权执行价
    quote['settle'] = fetch.Data[4]  # 结算价
    quote['data_source'] = 'Wind'
    quote['created_date'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(get_time(), key, len(quote), 'quotes to ' + quote['trade_date'][0].strftime('%Y-%m-%d') + '.')
    return quote


def get_quotes(options, start_date, end_date):
    print(get_time(), 'Download options quotes starts.')
    quote = []
    for key in options.keys():
        sym = options[key]
        print(get_time(), 'Download ' + key + ' starts.')
        quote.append(fetch_raw_quote(key, sym, start_date, end_date))
    print(get_time(), 'Download options quotes completes.')
    quote[0].to_excel('E:\\Projects\\volatility\\data\\sh50_quotes.xlsx')
    quote[1].to_excel('E:\\Projects\\volatility\\data\\sh300_quotes.xlsx')
    return quote


def calc_t_days(quote_50, basic_50):
    data_50 = pd.merge(quote_50, basic_50, how='left')
    data_50.drop(columns='trade_code', inplace=True)  # 删除trade_code列
    calls = data_50[data_50['call_or_put'] == u'认购'].copy()  # 分离出认购期权的数据
    calls.rename(columns={'settle': 'call'}, inplace=True)
    puts = data_50[data_50['call_or_put'] == u'认沽'].copy()  # 分离出认沽期权的数据
    puts.rename(columns={'settle': 'put'}, inplace=True)
    # 按交易日期、到期日、执行价为连接键合并，并删除重复数据
    options = pd.merge(calls, puts.loc[:, ['trade_date', 'expire_date', 'strike_price', 'put']],
                       on=['trade_date', 'expire_date', 'strike_price'], how='left').drop_duplicates()
    # 计算剩余到期的天数（自然日）
    options = options.loc[:, ['trade_date', 'expire_date', 'strike_price', 'call', 'put']]
    options['T_days'] = (options['expire_date'] - options['trade_date']).apply(lambda x: x.days)

    def small(x, s):
        '''确定第（s+1）小的值'''
        ts = list(set(x))
        ts.sort()
        return ts[s]

    last1 = options.groupby('trade_date')['T_days'].apply(small, s=0)  # 交易日第一短到期天数
    last2 = options.groupby('trade_date')['T_days'].apply(small, s=1)  # 交易日第二短到期天数
    last3 = options.groupby('trade_date')['T_days'].apply(small, s=2)  # 交易日第三短到期天数
    last = pd.concat([last1, last2, last3], axis=1)
    last.columns = ['last1', 'last2', 'last3']
    for i in range(len(last)):
        if last['last1'][i] <= 7:  # 近月合约到期期限必须不少于1个星期
            last.loc[last.index[i], 'last1'] = last.loc[last.index[i], 'last2']
            last.loc[last.index[i], 'last2'] = last.loc[last.index[i], 'last3']
    options = pd.merge(options, last, on='trade_date', how='left')
    return options


def fetch_shibor(start_date, end_date):
    check_connection()
    params = 'M0017141,M0017142'
    fetch = w.edb(params, start_date, end_date, 'Fill=Previous')
    shibor = pd.DataFrame()
    shibor['trade_date'] = fetch.Times
    shibor['shibor_1M'] = fetch.Data[0]
    shibor['shibor_3M'] = fetch.Data[1]
    shibor['data_source'] = 'Wind'
    shibor['created_date'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    shibor.set_index('trade_date', drop=False, inplace=True)
    print(get_time(), 'Shibor data acquired for', len(shibor['shibor_1M']), 'days.')
    return shibor


def data_slice(data, last_flag):
    """
    提取近月或次近月的期权数据
    Args:
        data:
            某一个交易的所有期权的数据
        last_flag:
            近月或次近月的标志，为'last1'或'last2'
    Returns:
        执行价格序列，看涨期权价格序列，看跌期权价格序列
    """
    data_last = data[data['T_days'] == data[last_flag]]
    data_last.set_index('strike_price', drop=False, inplace=True)
    return data_last['strike_price'], data_last['call'], data_last['put']


def vix(T_days, data, last_flag, r):
    """
    返回一个Vix实例
    """
    strikes, calls, puts = data_slice(data, last_flag)
    return Vix(T_days, strikes, calls, puts, r)


class Vix():
    """
    定义一个用方差互换和波动率互换方法编制VIX指数的类，基于50ETF期权
    """
    def __init__(self, T_days, strikes, calls, puts, r):
        """
        T_days:
            剩余到期天数
        strikes:
            当日可交易期权的执行价序列
        calls:
            看涨期权的价格序列
        puts:
            看跌期权的价格序列
        r:
            无风险利率，以3个月的Shibor为基准
        """
        self.T_days = T_days
        self.strikes = strikes
        self.calls = calls
        self.puts = puts
        self.r = r

    def T(self, Y_days=365):
        """
        计算剩余期限按年计算的比例
        Y_days:
            一年的总天数
        """
        return self.T_days / Y_days

    def F(self):
        """
        看涨期权和看跌期权价差绝对值最小对应的执行价格的远期价格
        """
        C_P = abs(self.calls - self.puts)
        K_F = C_P.idxmin()
        # print('认购认沽价差list：\n',  C_P)
        # print('最小认购认沽价差：\n', K_F)
        # print('远期价格：\n', K_F + (self.calls[K_F] - self.puts[K_F]) * exp(self.r * self.T()))
        return K_F + C_P[K_F] * exp(self.r * self.T())

    def Ks(self):
        """
        将执行价格从小到大排序，并返回列表
        """
        ks = list(self.strikes)
        ks.sort()
        # print('执行价格从小到大排序：\n', ks)
        return ks

    def K0(self):
        """
        低于F值的第一个执行价格为K0
        """
        ks = self.Ks()
        K_F = self.F()
        i = 0
        if ks[-1] < K_F:
            k0 = ks[-1]
        else:
            while ks[i] < K_F and i < len(ks):
                i += 1
            k0 = ks[i-1]
        # print('低于远期的第一个执行价格：\n', k0)
        return k0

    def delta_K(self):
        """
        执行价格的价差为相邻价格差的均值，[K(i+1) - K(i-1)] / 2
        特别地，最低执行价格的价差为K2 - K1，最高执行价格价差为K(N-1) - K(N-2)
        """
        delta_k = {}
        ks = self.Ks()
        delta_k[ks[0]] = ks[1] - ks[0]
        delta_k[ks[-1]] = ks[-1] - ks[-2]
        for i in range(1, len(ks) - 1):
            delta_k[ks[i]] = (ks[i+1] - ks[i-1]) / 2
        # print('执行价格价差 delta_k:\n', delta_k)
        return delta_k

    def Q_K(self):
        """
        期权费用，按结算价计算
        K < K0，为看跌期权价格，K > K0，为看涨期权价格，K = K0时是看涨和看跌期权价格的均值
        由于期权价格均大于0，无需考虑期权费为0的情况
        """
        q_k = {}
        k0 = self.K0()
        for k in self.Ks():
            if k < k0:
                q_k[k] = self.puts[k]
            elif k > k0:
                q_k[k] = self.calls[k]
            else:
                q_k[k] = (self.calls[k] + self.puts[k]) / 2
        # print('期权价格：\n', q_k)
        return q_k

    def sigma2(self):
        """
        计算某一个时刻，某一到期期限的波动率
        """
        delta_k = self.delta_K()
        q_k = self.Q_K()
        T = self.T()
        ks = self.Ks()
        sum_K = sum(delta_k[k] / k**2 * q_k[k] * exp(self.r * T)
                    for k in ks)
        return 2*sum_K / T - (self.F()/self.K0() - 1) ** 2 / T

    def volatility(self, other, M_days=30, Y_days=365):
        """
        将近月和次近月期权的波动率加权，得到对未来30天的波动率的预期值
        M_days:
            一个月的总天数
        Y_days:
            一年的总天数
        """
        last1 = self.sigma2() * self.T() * (other.T_days - M_days)\
                / (other.T_days - self.T_days)
        last2 = other.sigma2() * other.T() * (M_days - self.T_days)\
                / (other.T_days - self.T_days)
        # if self.T_days < 30:
        #     vix_num = 100 * sqrt((last1 + last2) * Y_days / M_days)
        # else:
        #     vix_num = 100 * sqrt(last1)
        # print('近月波动率：', last1, '次近月波动率：', last2)
        vix_num = 100 * sqrt((last1 + last2) * Y_days / M_days)
        # print('近月合约剩余天数：', self.T_days, '\tVIX 指数：', vix_num)
        return vix_num

    def skew_idv(self):
        k0 = self.K0()
        f0 = self.F()
        delta_k = self.delta_K()
        q_k = self.Q_K()
        T = self.T()
        ks = self.Ks()
        ep1 = -(1 + log(f0/k0) - f0/k0)
        ep2 = 2*log(k0/f0)*(f0/k0 - 1) + 0.5*(log(k0/f0))**2
        ep3 = 3*(log(k0/f0))**2*(1/3*log(k0/f0) - 1 + f0/k0)
        p1 = -exp(self.r * T) * sum(q_k[k] * delta_k[k] / k**2 for k in ks) + ep1
        p2 = exp(self.r * T) * sum(2 * q_k[k] * delta_k[k] / k**2 * (1 - log(k/f0)) for k in ks) + ep2
        p3 = exp(self.r * T) * sum(3 * q_k[k] * delta_k[k] / k**2 * (2 * log(k/f0) - (log(k/f0)) ** 2)
                                   for k in ks) + ep3
        skew_idv = (p3 - 3 * p1 * p2 + 2 * p1 ** 3) / (p2 - p1 ** 2)**(3/2)
        return skew_idv

    def skew(self, other, M_days=30):
        s1 = self.skew_idv()
        s2 = other.skew_idv()
        w = (other.T_days - M_days) / (other.T_days - self.T_days)
        skew_num = 100 - 10 * (w * s1 + (1 - w) * s2)
        return skew_num


def calc_vix_skew(options_50, shibor):
    data = options_50
    dates = data['trade_date'].drop_duplicates()
    data.set_index(['trade_date', 'expire_date'], drop=False, inplace=True)

    VIX = {}  # 将得到的volatility值，以日期为键，保存到字典VIX
    SKEW = {}
    for datei in dates:  # quote 得到日期格式为 timestamp.Timestamp
        date = datei.to_pydatetime().date()  # shibor 得到日期格式为 datetime.date
        r = shibor.loc[date, 'shibor_3M']/100  # shibor 单位为%，处理后再使用
        data_t = data.xs(date, level=0)  # 某一交易日的数据
        T_days_1 = data_t['last1'][0]  # 近月的剩余到期天数
        T_days_2 = data_t['last2'][0]  # 次近月的剩余到期天数
        vix_1 = vix(T_days_1, data_t, 'last1', r)
        vix_2 = vix(T_days_2, data_t, 'last2', r)
        VIX[date] = vix_1.volatility(vix_2)
        SKEW[date] = vix_1.skew(vix_2)
    return VIX, SKEW


def main():
    check_connection()
    options = {u'沪50ETF': '510050.SH', u'沪300ETF': '510300.SH'}  # 深交所300ETF还无法自动取数
    # key = u'沪50ETF'
    # sym = '510050.SH'
    start_date = '2020-09-18'
    end_date = '2020-10-12'

    basic = get_contract(options, end_date)  # 获取合约信息
    basic_50 = basic[0].loc[:, ['option_code', 'trade_code', 'call_or_put',
                                'strike_price', 'expire_date']]
    basic_300 = basic[1].loc[:, ['option_code', 'trade_code', 'call_or_put',
                                 'strike_price', 'expire_date']]

    quote = get_quotes(options, start_date, end_date)  # 获取期权交易价格信息
    quote_50 = quote[0].loc[:, ['trade_date', 'trade_code', 'option_code', 'settle']]
    quote_300 = quote[1].loc[:, ['trade_date', 'trade_code', 'option_code', 'settle']]

    options_50 = calc_t_days(quote_50, basic_50)  # 计算期权到期日，以备加权用
    options_50.to_excel('E:\\Projects\\volatility\\data\\50_options.xlsx')
    options_300 = calc_t_days(quote_300, basic_300)
    options_300.to_excel('E:\\Projects\\volatility\\data\\300_options.xlsx')

    shibor = fetch_shibor(start_date, end_date)  # 获取无风险利率
    shibor.to_excel('E:\\Projects\\volatility\\data\\shibor_opts.xlsx')
    close_connection()

    print(get_time(), 'iVX50 calculation starts.')
    VIX_50, SKEW_50 = calc_vix_skew(options_50, shibor)
    VIX_50 = pd.Series(VIX_50).sort_index()
    SKEW_50 = pd.Series(SKEW_50).sort_index()
    print(get_time(), 'iVX50 calculation ends.')

    print(get_time(), 'iVX300 calculation starts.')
    VIX_300, SKEW_300 = calc_vix_skew(options_300, shibor)
    VIX_300 = pd.Series(VIX_300).sort_index()
    SKEW_300 = pd.Series(SKEW_300).sort_index()
    print(get_time(), 'iVX300 calculation ends.')

    def description(VIX_50, SKEW_50, sec_str):
        month_d = str(VIX_50.index[-1].month)
        day_d = str(VIX_50.index[-1].day)
        change1 = '{0:.2f}'.format((VIX_50[-1]/VIX_50[-2] - 1) * 100)
        change2 = '{0:.2f}'.format((SKEW_50[-1]/SKEW_50[-2] - 1) * 100)
        end_value1 = '{0:.2f}'.format(VIX_50[-1])
        end_value2 = '{0:.2f}'.format(SKEW_50[-1])
        print(month_d+u'月'+day_d+u'日，iVX'+sec_str+'上涨下跌'+change1+'%至'
              +end_value1+'；Skew'+sec_str+'上涨下跌'+change2+'%至'+end_value2+'。')

    description(VIX_50, SKEW_50, '50')
    description(VIX_300, SKEW_300, '300')

    with pd.ExcelWriter('E:\\Projects\\volatility\\data\\raw_iVX_SKEW.xlsx')\
            as writer:
        VIX_50.to_excel(writer)
        SKEW_50.to_excel(writer, startcol=3)
        VIX_300.to_excel(writer, startcol=6)
        SKEW_300.to_excel(writer, startcol=9)


if __name__ == '__main__':
    import time
    main()
