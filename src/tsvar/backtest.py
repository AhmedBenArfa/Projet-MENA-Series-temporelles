import numpy as np
from scipy import stats

def violations(y_true, var_levels) -> np.ndarray:
    return np.asarray(y_true) < np.asarray(var_levels)

def _safe_log(x):
    return np.log(x) if x > 0 else 0.0

def kupiec_pof(n_viol, n_obs, alpha) -> dict:
    pi = n_viol/n_obs
    ll_null = (n_obs-n_viol)*_safe_log(1-alpha) + n_viol*_safe_log(alpha)
    ll_alt  = (n_obs-n_viol)*_safe_log(1-pi)   + n_viol*_safe_log(pi)
    lr = -2*(ll_null-ll_alt)
    p = 1-stats.chi2.cdf(lr,1)
    return {"LR":lr,"pvalue":p,"reject":bool(p<0.05)}

def christoffersen(viol, alpha) -> dict:
    v = np.asarray(viol).astype(int)
    n00=n01=n10=n11=0
    for a,b in zip(v[:-1],v[1:]):
        if a==0 and b==0: n00+=1
        elif a==0 and b==1: n01+=1
        elif a==1 and b==0: n10+=1
        else: n11+=1
    pi=(n01+n11)/max(n00+n01+n10+n11,1)
    pi0=n01/max(n00+n01,1); pi1=n11/max(n10+n11,1)
    ll_ind = (n00+n10)*_safe_log(1-pi)+(n01+n11)*_safe_log(pi)
    ll_alt = n00*_safe_log(1-pi0)+n01*_safe_log(pi0)+n10*_safe_log(1-pi1)+n11*_safe_log(pi1)
    lr_ind = -2*(ll_ind-ll_alt)
    n_viol=int(v.sum()); n=len(v)
    lr_pof = kupiec_pof(n_viol,n,alpha)["LR"]
    lr_cc = lr_pof+lr_ind
    p_cc = 1-stats.chi2.cdf(lr_cc,2)
    return {"LR_ind":lr_ind,"LR_cc":lr_cc,
            "pvalue_cc":p_cc,"reject":bool(p_cc<0.05)}

def basel_zone(n_viol, n_obs, alpha) -> str:
    scaled = n_viol*(250/n_obs)                 # normalize to 250-day window
    if alpha==0.01:
        return "green" if scaled<=4 else "yellow" if scaled<=9 else "red"
    # alpha=0.05: heuristic bands (not an official Basel traffic-light table; official table defined at 99%)
    return "green" if scaled<=17 else "yellow" if scaled<=25 else "red"

def backtest_summary(y_true, var_levels, alpha) -> dict:
    v = violations(y_true, var_levels); n=len(v); k=int(v.sum())
    return {"n":n,"n_violations":k,"expected_rate":alpha,"observed_rate":k/n,
            "kupiec":kupiec_pof(k,n,alpha),
            "christoffersen":christoffersen(v,alpha),
            "basel_zone":basel_zone(k,n,alpha)}
