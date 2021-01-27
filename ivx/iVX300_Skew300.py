# -*- coding: utf-8 -*-

import pandas as pd
from math import exp, sqrt, log


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
        K_F = C_P.astype('float64').idxmin()
        return K_F + C_P[K_F] * exp(self.r * self.T())

    def Ks(self):
        """
        将执行价格从小到大排序，并返回列表
        """
        ks = list(self.strikes)
        ks.sort()
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
        vix_num = 100 * sqrt((last1 + last2) * Y_days / M_days)
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

def main():
    # import basic contract info
    basic = pd.read_excel('C:\\Projects\\volatility\\data\\沪300ETF期权合约基本资料.xlsx', parse_date=True)
    basic = basic.loc[:, ['trade_code', 'type', 'expire']]
    print('contracts info loaded.')

    # import daily quotes
    data2020 = pd.read_excel('C:\\Projects\\volatility\\data\\沪300ETF期权日行情2019-2020.xlsx', parse_date=True)
    data2020 = data2020.loc[:, ['date', 'trade_code', 'strike', 'settle']]
    data = data2020
    print('data concatenated.')

    # separate call and put options
    datas = pd.merge(data, basic, how='left')  # 按trade_code合并
    datas.drop(columns='trade_code', inplace=True)  # 删除trade_code列
    calls = datas[datas['type'] == '认购'].copy()  # 分离出认购期权的数据
    puts = datas[datas['type'] == '认沽'].copy()  # 分离出认沽期权的数据
    calls.rename(columns={'settle': 'call'}, inplace=True)
    puts.rename(columns={'settle': 'put'}, inplace=True)

    # 按交易日期、到期日、执行价为连接键合并，并删除重复数据
    options = pd.merge(calls, puts.loc[:, ['date', 'expire', 'strike', 'put']],
                       on=['date', 'expire', 'strike'], how='left').drop_duplicates()
    # 计算剩余到期的天数（自然日）
    options = options.loc[:, ['date', 'expire', 'strike', 'call', 'put']]
    options['T_days'] = (options['expire'] - options['date']).apply(lambda x: x.days)

    def small(x, s):
        '''确定第（s+1）小的值'''
        ts = list(set(x))
        ts.sort()
        return ts[s]

    last1 = options.groupby('date')['T_days'].apply(small, s=0)  # 交易日第一短到期天数
    last2 = options.groupby('date')['T_days'].apply(small, s=1)  # 交易日第二短到期天数
    last3 = options.groupby('date')['T_days'].apply(small, s=2)  # 交易日第三短到期天数
    last = pd.concat([last1, last2, last3], axis=1)
    last.columns = ['last1', 'last2', 'last3']
    for i in range(len(last)):
        if last['last1'][i] <= 7:  # 近月合约到期期限必须不少于1个星期
            last.loc[last.index[i], 'last1'] = last.loc[last.index[i], 'last2']
            last.loc[last.index[i], 'last2'] = last.loc[last.index[i], 'last3']
    options = pd.merge(options, last, on='date', how='left')

    shibor = pd.read_excel('C:\\Projects\\volatility\\data\\shibor.xlsx', index_col='date')
    data = options
    dates = data['date'].drop_duplicates()
    data.set_index(['date', 'expire'], drop=False, inplace=True)

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
        data_last.set_index('strike', drop=False, inplace=True)
        return data_last['strike'], data_last['call'], data_last['put']

    def vix(T_days, data, last_flag, r):
        """
        返回一个Vix实例
        """
        strikes, calls, puts = data_slice(data, last_flag)
        return Vix(T_days, strikes, calls, puts, r)

    VIX = {}  # 将得到的volatility值，以日期为键，保存到字典VIX
    SKEW = {}
    for date in dates:
        print(date)
        r = shibor.loc[date, 'shibor_3M']
        data_t = data.xs(date, level=0)  # 某一交易日的数据
        T_days_1 = data_t['last1'][0]  # 近月的剩余到期天数
        T_days_2 = data_t['last2'][0]  # 次近月的剩余到期天数
        vix_1 = vix(T_days_1, data_t, 'last1', r)
        vix_2 = vix(T_days_2, data_t, 'last2', r)
        VIX[date] = vix_1.volatility(vix_2)
        SKEW[date] = vix_1.skew(vix_2)

    VIX = pd.Series(VIX).sort_index()
    VIX.to_excel('C:\\Projects\\volatility\\data\\raw_沪300_VIX.xlsx')

    SKEW = pd.Series(SKEW).sort_index()
    SKEW.to_excel('C:\\Projects\\volatility\\data\\raw_沪300_SKEW.xlsx')


if __name__ == '__main__':
    import pandas as pd
    main()
