# -*- coding: utf-8 -*-

from jqdatasdk import *
from datetime import datetime
import pandas as pd
import time


def get_time():
    # 获取当前时间
    return time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime(time.time()))


def check_query_count():
    # 向 JoinQuant 服务器查询当日数据配额使用量及剩余量
    query_count = get_query_count()
    print("Used %d, spare %d." %(query_count['total'] - query_count['spare'], query_count['spare']))
    return


def get_contract(code):
    print(get_time(), u'期权合约数据开始下载。')
    contracts_fetch = pd.DataFrame()
    # 查询当前可交易的50ETF期权合约信息
    df1 = opt.run_query(query(opt.OPT_CONTRACT_INFO).filter((opt.OPT_CONTRACT_INFO.underlying_symbol == code)
                                                            & (opt.OPT_CONTRACT_INFO.last_trade_date > datetime.now())))
    # contracts_fetch['option_code'] = df1['code']  # 期权合约代码
    contracts_fetch['trade_code'] = df1['trading_code']  # 合约交易代码
    contracts_fetch['call_or_put'] = df1['contract_type']  # 合约认沽认购类型
    contracts_fetch['strike_price'] = df1['exercise_price']  # 行权价
    contracts_fetch['expire_date'] = df1['expire_date']  # 到期日期
    print(get_time(), u'期权', code, u'已下载', len(contracts_fetch), u'个合约，日期为' +
          contracts_fetch['expire_date'][len(contracts_fetch) - 1].strftime('%Y-%m-%d') + '。')
    return contracts_fetch


def get_quotes(contracts_current, date_today):
    # 获取指定交易日的期权合约成交数据
    raw_quote = pd.DataFrame()
    for contract in contracts_current:
        # print(get_time(), u'正在下载合约', contract, '日行情数据。')
        df2 = opt.run_query(query(opt.OPT_DAILY_PRICE).filter((opt.OPT_DAILY_PRICE.code == contract)
                                                                    & (opt.OPT_DAILY_PRICE.date == date_today)))
        if raw_quote.empty:
            raw_quote = df2
        else:
            raw_quote = pd.merge(raw_quote, df2, how='outer')
    raw_quote = raw_quote.loc[:, ['code', 'date', 'settle_price']]
    raw_quote.rename(columns={'code': 'option_code'}, inplace=True)
    raw_quote.rename(columns={'date': 'trade_date'}, inplace=True)
    return raw_quote


def small(x, s):
    '''确定第（s+1）小的值'''
    ts = list(set(x))
    ts.sort()
    return ts[s]


def calc_remaining_days(contracts_50, quotes_50):
    data_50 = pd.merge(quotes_50, contracts_50, how='left')
    # data_50.drop(columns='trade_code', inplace=True)  # 删除trade_code列
    # 分离认购期权数据
    calls = data_50[data_50['call_or_put'] == 'CO'].copy()
    calls.rename(columns={'settle_price': 'call'}, inplace=True)
    # 分离认沽期权数据
    puts = data_50[data_50['call_or_put'] == 'PO'].copy()
    puts.rename(columns={'settle_price': 'put'}, inplace=True)
    # 按交易日期、到期日、执行价为连接键合并，并删除重复数据
    options_merge = pd.merge(calls, puts.loc[:, ['trade_date', 'expire_date', 'strike_price', 'put']],
                       on=['trade_date', 'expire_date', 'strike_price'], how='left').drop_duplicates()
    # 计算剩余到期的天数（自然日）
    options_merge = options_merge.loc[:, ['trade_date', 'expire_date', 'strike_price', 'call', 'put']]
    options_merge['T_days'] = (options_merge['expire_date'] - options_merge['trade_date']).apply(lambda x: x.days)
    last1 = options_merge.groupby('trade_date')['T_days'].apply(small, s=0)  # 交易日第一短到期天数
    last2 = options_merge.groupby('trade_date')['T_days'].apply(small, s=1)  # 交易日第二短到期天数
    last3 = options_merge.groupby('trade_date')['T_days'].apply(small, s=2)  # 交易日第三短到期天数
    last = pd.concat([last1, last2, last3], axis=1)
    last.columns = ['last1', 'last2', 'last3']
    for i in range(len(last)):
        if last['last1'][i] <= 7:  # 近月合约到期期限必须不少于1个星期
            last.loc[last.index[i], 'last1'] = last.loc[last.index[i], 'last2']
            last.loc[last.index[i], 'last2'] = last.loc[last.index[i], 'last3']
    options_merge = pd.merge(options_merge, last, on='trade_date', how='left')
    return options_merge


def fetch_shibor(fetch_date):
    # 获取指定日期的 shibor 数据
    shibor_fetch = pd.DataFrame()
    df3 = macro.run_query(query(macro.MAC_LEND_RATE).filter((macro.MAC_LEND_RATE.day==fetch_date)
                                                            & (macro.MAC_LEND_RATE.market_id=='5')
                                                            & (macro.MAC_LEND_RATE.term_id=='3')).limit(10))
    shibor_fetch['trade_date'] = df3['day']
    shibor_fetch['shibor_3M'] = df3['interest_rate']
    return shibor_fetch


def main():
    # 身份认证
    id_jq = ''
    pass_jq = ''
    auth(id_jq, pass_jq)

    # 查询是否连接成功
    auth_status = is_auth()
    print(auth_status)

    # 查看数据配额
    check_query_count()

    # 期权种类
    # 上交所上证50ETF期权(标的为华夏上证50ETF,代码510050)
    # code = "510050.XSHG"
    # 上交所沪深300ETF期权(标的为华泰柏瑞沪深300ETF,代码510300)
    # code = "510300.XSHG"
    # 深交所沪深300ETF期权(标的为嘉实沪深300ETF,代码159919)
    # code = "159919.XSHE"
    # 中金所沪深300指数期权
    # code = "000300.CCFX"
    # options = {u'沪50ETF': '510050.XSHG', u'沪300ETF': '510300.XSHG', u'深300ETF': '159919.XSHE',
    #            u'金300ETF': '000300.CCFX'}  # 调试完成后启用其他期权
    options = {u'沪50ETF': '510050.XSHG'}

    # 获取期权合约信息
    contracts = {}
    for key in options:
        print(get_time(), u'开始下载' + key + u'期权合约。')
        contracts[key] = get_contract(options[key])

    # 获取期权日行情信息
    quotes = {}
    # date_today = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    date_today = '2021-02-01'
    for key in options:
        print(get_time(), u'开始下载' + key + u'日行情。')
        quotes[key] = get_quotes(contracts[key]['option_code'], date_today)

    # 计算期权到期日
    options_combined = {}
    for key in options:
        print(get_time(), u'开始计算' + key + u'期权到期日。')
        options_combined[key] = calc_remaining_days(contracts[key], quotes[key])

    # 获取无风险利率
    shibor = fetch_shibor(date_today)

    # 计算波动率指数 iVX


if __name__ == '__main__':
    main()

