#dynamic tangent portfolio
import ccxt
from math import *
import pandas as pd
import numpy as np
import cvxopt as opt
import cvxopt.solvers as optsolvers
import warnings
from operator import itemgetter
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
from concurrent.futures import *
import random
import pickle
"""PortfolioOpt: Financial Portfolio Optimization

This module provides a set of functions for financial portfolio
optimization, such as construction of Markowitz portfolios, minimum
variance portfolios and tangency portfolios (i.e. maximum Sharpe ratio
portfolios) in Python. The construction of long-only, long/short and
market neutral portfolios is supported."""


__all__ = ['markowitz_portfolio',
           'min_var_portfolio',
           'tangency_portfolio',
           'max_ret_portfolio',
           'truncate_weights']


def markowitz_portfolio(cov_mat, exp_rets, target_ret,
                        allow_short=False, market_neutral=False):
    """
    Computes a Markowitz portfolio.

    Parameters
    ----------
    cov_mat: pandas.DataFrame
        Covariance matrix of asset returns.
    exp_rets: pandas.Series
        Expected asset returns (often historical returns).
    target_ret: float
        Target return of portfolio.
    allow_short: bool, optional
        If 'False' construct a long-only portfolio.
        If 'True' allow shorting, i.e. negative weights.
    market_neutral: bool, optional
        If 'False' sum of weights equals one.
        If 'True' sum of weights equal zero, i.e. create a
            market neutral portfolio (implies allow_short=True).

    Returns
    -------
    weights: pandas.Series
        Optimal asset weights.
    """
    if not isinstance(cov_mat, pd.DataFrame):
        raise ValueError("Covariance matrix is not a DataFrame")

    if not isinstance(exp_rets, pd.Series):
        raise ValueError("Expected returns is not a Series")

    if not isinstance(target_ret, float):
        raise ValueError("Target return is not a float")

    if not cov_mat.index.equals(exp_rets.index):
        raise ValueError("Indices do not match")

    if market_neutral and not allow_short:
        warnings.warn("A market neutral portfolio implies shorting")
        allow_short=True

    n = len(cov_mat)

    P = opt.matrix(cov_mat.values)
    q = opt.matrix(0.0, (n, 1))

    # Constraints Gx <= h
    if not allow_short:
        # exp_rets*x >= target_ret and x >= 0
        G = opt.matrix(np.vstack((-exp_rets.values,
                                  -np.identity(n))))
        h = opt.matrix(np.vstack((-target_ret,
                                  +np.zeros((n, 1)))))
    else:
        # exp_rets*x >= target_ret
        G = opt.matrix(-exp_rets.values).T
        h = opt.matrix(-target_ret)

    # Constraints Ax = b
    # sum(x) = 1
    A = opt.matrix(1.0, (1, n))

    if not market_neutral:
        b = opt.matrix(1.0)
    else:
        b = opt.matrix(0.0)

    # Solve
    optsolvers.options['show_progress'] = False
    sol = optsolvers.qp(P, q, G, h, A, b)

    if sol['status'] != 'optimal':
        warnings.warn("Convergence problem")

    # Put weights into a labeled series
    weights = pd.Series(sol['x'], index=cov_mat.index)
    return weights


def min_var_portfolio(cov_mat, allow_short=False):
    """
    Computes the minimum variance portfolio.

    Note: As the variance is not invariant with respect
    to leverage, it is not possible to construct non-trivial
    market neutral minimum variance portfolios. This is because
    the variance approaches zero with decreasing leverage,
    i.e. the market neutral portfolio with minimum variance
    is not invested at all.

    Parameters
    ----------
    cov_mat: pandas.DataFrame
        Covariance matrix of asset returns.
    allow_short: bool, optional
        If 'False' construct a long-only portfolio.
        If 'True' allow shorting, i.e. negative weights.

    Returns
    -------
    weights: pandas.Series
        Optimal asset weights.
    """
    if not isinstance(cov_mat, pd.DataFrame):
        raise ValueError("Covariance matrix is not a DataFrame")

    n = len(cov_mat)

    P = opt.matrix(cov_mat.values)
    q = opt.matrix(0.0, (n, 1))

    # Constraints Gx <= h
    if not allow_short:
        # x >= 0
        G = opt.matrix(-np.identity(n))
        h = opt.matrix(0.0, (n, 1))
    else:
        G = None
        h = None

    # Constraints Ax = b
    # sum(x) = 1
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Solve
    optsolvers.options['show_progress'] = False
    sol = optsolvers.qp(P, q, G, h, A, b)

    if sol['status'] != 'optimal':
        warnings.warn("Convergence problem")

    # Put weights into a labeled series
    weights = pd.Series(sol['x'], index=cov_mat.index)
    return weights


def tangency_portfolio(cov_mat, exp_rets, allow_short=False):
    """
    Computes a tangency portfolio, i.e. a maximum Sharpe ratio portfolio.

    Note: As the Sharpe ratio is not invariant with respect
    to leverage, it is not possible to construct non-trivial
    market neutral tangency portfolios. This is because for
    a positive initial Sharpe ratio the sharpe grows unbound
    with increasing leverage.

    Parameters
    ----------
    cov_mat: pandas.DataFrame
        Covariance matrix of asset returns.
    exp_rets: pandas.Series
        Expected asset returns (often historical returns).
    allow_short: bool, optional
        If 'False' construct a long-only portfolio.
        If 'True' allow shorting, i.e. negative weights.

    Returns
    -------
    weights: pandas.Series
        Optimal asset weights.
    """
    if not isinstance(cov_mat, pd.DataFrame):
        raise ValueError("Covariance matrix is not a DataFrame")

    if not isinstance(exp_rets, pd.Series):
        raise ValueError("Expected returns is not a Series")

    if not cov_mat.index.equals(exp_rets.index):
        raise ValueError("Indices do not match")

    n = len(cov_mat)

    P = opt.matrix(cov_mat.values)
    q = opt.matrix(0.0, (n, 1))

    # Constraints Gx <= h
    if not allow_short:
        # exp_rets*x >= 1 and x >= 0
        G = opt.matrix(np.vstack((-exp_rets.values,
                                  -np.identity(n))))
        h = opt.matrix(np.vstack((-1.0,
                                  np.zeros((n, 1)))))
    else:
        # exp_rets*x >= 1
        G = opt.matrix(-exp_rets.values).T
        h = opt.matrix(-1.0)

    # Solve
    optsolvers.options['show_progress'] = False
    sol = optsolvers.qp(P, q, G, h)

    if sol['status'] != 'optimal':
        warnings.warn("Convergence problem")

    # Put weights into a labeled series
    weights = pd.Series(sol['x'], index=cov_mat.index)

    # Rescale weights, so that sum(weights) = 1
    weights /= weights.sum()
    return weights


def max_ret_portfolio(exp_rets):
    """
    Computes a long-only maximum return portfolio, i.e. selects
    the assets with maximal return. If there is more than one
    asset with maximal return, equally weight all of them.

    Parameters
    ----------
    exp_rets: pandas.Series
        Expected asset returns (often historical returns).

    Returns
    -------
    weights: pandas.Series
        Optimal asset weights.
    """
    if not isinstance(exp_rets, pd.Series):
        raise ValueError("Expected returns is not a Series")

    weights = exp_rets[:]
    weights[weights == weights.max()] = 1.0
    weights[weights != weights.max()] = 0.0
    weights /= weights.sum()

    return weights


def truncate_weights(weights, min_weight=0.01, rescale=True):
    """
    Truncates small weight vectors, i.e. sets weights below a treshold to zero.
    This can be helpful to remove portfolio weights, which are negligibly small.

    Parameters
    ----------
    weights: pandas.Series
        Optimal asset weights.
    min_weight: float, optional
        All weights, for which the absolute value is smaller
        than this parameter will be set to zero.
    rescale: boolean, optional
        If 'True', rescale weights so that weights.sum() == 1.
        If 'False', do not rescale.

    Returns
    -------
    adj_weights: pandas.Series
        Adjusted weights.
    """
    if not isinstance(weights, pd.Series):
        raise ValueError("Weight vector is not a Series")

    adj_weights = weights[:]
    adj_weights[adj_weights.abs() < min_weight] = 0.0

    if rescale:
        if not adj_weights.sum():
            raise ValueError("Cannot rescale weight vector as sum is not finite")

        adj_weights /= adj_weights.sum()

    return adj_weights
def main(base_currency, interval="5m", max_portfolio_num=10):
    max_portfolio_num = int(max_portfolio_num)
    pd.options.display.precision = 8
    binance = ccxt.binance()
    quote = base_currency
    ranking = list()
    tickers = dict()
    tickers["binance"] = binance.fetch_tickers()
    exchange_info = binance.publicGetExchangeinfo()["symbols"]
    digits = dict()
    for i in exchange_info:
        digits[i["symbol"]] = dict()
        digits[i["symbol"]]["amount"] = int(-log10(float(i['filters'][2]['stepSize'])))
        digits[i["symbol"]]["price"] = int(-log10(float(i['filters'][0]['tickSize'])))
        digits[i["symbol"]]["min_amount"] = float(i['filters'][3]['minNotional'])
    for ticker in tickers["binance"]:
        if ticker.split('/')[1] == quote:
            ranking.append([ticker, tickers["binance"][ticker]['quoteVolume']])
    ohlcv = dict()
    data_tickers = [["symbol","bid","ask"]]
    for ticker in tickers["binance"]:
        if "bid" in tickers["binance"][ticker]:
            tmp = [ticker, tickers["binance"][ticker]["bid"], tickers["binance"][ticker]["ask"]]
            data_tickers.append(tmp)
    executor = ThreadPoolExecutor(max_workers = 2)
    for r in ranking[:100]:
        ohlcv[r[0]] = executor.submit(binance.fetch_ohlcv, r[0], interval)
    for x in ohlcv:
        ohlcv[x] = ohlcv[x].result()
    close = dict()
    for i in ohlcv:
        close[i] = np.array(ohlcv[i])[:,4]
    dataset = pd.DataFrame(close).pct_change().dropna()
    avg_rets = dataset.mean()
    cov = dataset.cov()
    weights = tangency_portfolio(cov, avg_rets)
    weights = weights.sort_values(ascending=False)[:max_portfolio_num]
    weights /= np.sum(weights)
    print(weights)
    plt.pie(weights,
            autopct="%1.1f%%",
            labels=weights.index)
    total_ret = 0
    portfolio_variance = 0
    c = cov.values
    for w in range(len(weights)):
        total_ret += weights[w]*avg_rets[w]
        for v in range(len(weights)):
            portfolio_variance +=  weights[w]*weights[v]*c[w,v]
    print("total_return : {}".format(total_ret))
    print("portfolio variance : {}".format(portfolio_variance))
    span = int(interval[0])
    daily_implied_return = (1+total_ret)**(60/span*24)-1
    daily_implied_std = sqrt(portfolio_variance)*sqrt(60/span*24)
    print("daily implied return : {}".format(daily_implied_return))
    ob = dict()
    last_value = dict()
    number_of_colors = len(weights)
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    new_weights = list()
    for x, y in weights.items():
        new_weights.append({"symbol":x, "value":y})
    result = {"num_of_length":len(weights.tolist()),"total_ret":total_ret, "daily_implied_return":daily_implied_return, "weights":new_weights, "portfolio_std":daily_implied_std, "label":weights.index.tolist(), "data":weights.tolist(),"color":color}
    with open("hello/static/result.json","w") as f:
        print(json.dump(result, f))
        print("writing completed")
    print(result)
    return result
