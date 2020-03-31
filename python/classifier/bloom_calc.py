import math

#NOTE m here is the number of bits in the array, not the number of slots in the array
"""
n - number of items in the filter
p - number of false positives
m - number of bits in the filter
k - number of hash functinos
this logic was migrated from Thomas Hurst https://hur.st/bloomfilter/ to python
"""


def _p_from_kr(k, r):
    return math.pow(1 - math.exp(-k / r), k)


def _k_from_r(r):
    return round(math.log(2) * r)


def _r_from_pk(p, k):
    return -k / math.log(1 - math.exp(math.log(p) / k))


def _r_from_mn(m, n):
    return m / n


def km_from_np(n, p):
    m = math.ceil(n * math.log(p) / math.log(1 / math.pow(2, math.log(2))))

    r = _r_from_mn(m, n)
    k = _k_from_r(r)
    p = _p_from_kr(k, r)

    return (k, m, n, p)


def kp_from_mn(m, n):
    r = _r_from_mn(m, n)
    k = _k_from_r(r)
    p = _p_from_kr(k, r)

    return (k, m, n, p)


def kn_from_mp(m, p):
    n = math.ceil((m * math.log(math.pow(1 / 2, math.log(2))) / math.log(p)))
    r = _r_from_mn(m, n)
    k = _k_from_r(r)
    p = _p_from_kr(k, r)

    return (k, m, n, p)


def p_from_kmn(k, m, n):
    p = _p_from_kr(k, _r_from_mn(m, n))
    return (k, m, n, p)


def n_from_kmp(k, m, p):
    r = _r_from_pk(p, k)
    n = math.ceil(m / r)

    return (k, m, n, p)


def m_from_knp(k, n, p):
    r = _r_from_pk(p, k)
    m = math.ceil(n * r)

    return (k, m, n, p)


def k_from_mnp(m, n, p):
    r = _r_from_mn(m, n)

    opt_k = -round(math.log2(p))

    # TODO
    raise Exception("not yet implementaed")
