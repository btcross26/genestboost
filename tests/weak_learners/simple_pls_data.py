"""
Dummy data for SimplePLS unit tests. The X-matrix is a randomly generated
IID normal matrix. The y-vector is then generated as
X[:,0] * 4 + X[:,3] * 2 - X[:,7] * 6 and has standard normal variabe noise
added.
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-12-20


import numpy as np

X = np.array(
    [
        [
            0.7497189,
            -1.06018418,
            -0.11032608,
            0.78438204,
            0.19260947,
            0.50298249,
            1.5294589,
            -0.05471981,
        ],
        [
            1.06347668,
            -0.40624225,
            -0.81178313,
            1.07106614,
            -0.16331986,
            0.70117137,
            0.06280904,
            -2.13638837,
        ],
        [
            -0.23012922,
            1.24230194,
            0.08023191,
            -0.56656563,
            0.54415714,
            -0.42935247,
            -0.65038073,
            -0.28894214,
        ],
        [
            -0.29235792,
            0.2294957,
            1.93774459,
            -1.20879714,
            -0.95673131,
            -0.44558123,
            -0.30900366,
            0.90279066,
        ],
        [
            -0.57345769,
            -0.10583596,
            1.05813445,
            -0.90656399,
            -0.20535915,
            0.4088877,
            -0.29322562,
            0.46461794,
        ],
        [
            -0.60320655,
            0.92424305,
            -1.61841182,
            -0.32477063,
            -2.81843102,
            0.29488012,
            1.43215578,
            0.64665655,
        ],
        [
            0.43941082,
            0.34926137,
            -0.21402346,
            -0.70259443,
            1.89322329,
            -0.039443,
            0.34830967,
            -0.24299307,
        ],
        [
            0.0542351,
            1.90231971,
            0.19827739,
            -0.47499584,
            -0.04715391,
            0.37624003,
            -0.1521753,
            0.74272243,
        ],
        [
            -0.83777646,
            -0.60493518,
            -0.36978125,
            -0.38274495,
            1.46338802,
            -1.48765481,
            -0.46621875,
            2.00852844,
        ],
        [
            1.27372897,
            -0.15266175,
            -0.90108836,
            0.79228567,
            0.61864826,
            0.02863394,
            -2.25454165,
            -0.40520212,
        ],
        [
            -0.8762165,
            -1.51913904,
            -0.89146386,
            -0.48113659,
            -0.32069465,
            1.90717444,
            0.48954002,
            1.35059104,
        ],
        [
            0.39394328,
            1.08403074,
            1.83289211,
            0.58108581,
            0.14192728,
            -0.57282836,
            0.54046339,
            -0.72882989,
        ],
        [
            -0.93070293,
            0.00487264,
            0.59144336,
            0.39625409,
            -2.01781745,
            0.61724187,
            -0.43935799,
            1.1413281,
        ],
        [
            0.62422156,
            -0.20650372,
            0.93015685,
            0.19988466,
            1.68635126,
            0.19102748,
            0.76550233,
            1.10045332,
        ],
        [
            -0.92175162,
            0.25588025,
            0.70790095,
            -0.7504252,
            -0.15624115,
            0.01268147,
            -0.36518843,
            -0.8423974,
        ],
        [
            -1.39389685,
            1.02772955,
            -0.43915297,
            -0.89316328,
            0.37577078,
            -0.23811136,
            -1.45823825,
            1.56511203,
        ],
        [
            -1.26510065,
            -0.80747455,
            1.3452855,
            0.74963613,
            0.43589739,
            0.77307508,
            1.27978188,
            0.39215796,
        ],
        [
            -1.02384098,
            0.94920305,
            -1.05036418,
            -2.0545511,
            0.89267575,
            -1.13461524,
            2.19288867,
            -1.00493085,
        ],
        [
            1.25690434,
            0.88679479,
            -0.57027041,
            0.62897259,
            0.26111358,
            1.66233709,
            0.74892626,
            -0.06926899,
        ],
        [
            3.04035907,
            1.30557272,
            -0.06177561,
            -0.23502601,
            -0.00313698,
            0.45355249,
            1.07846669,
            -0.31694713,
        ],
    ]
)
y = np.array(
    [
        3.87264128,
        19.81146556,
        0.16966769,
        -10.0074944,
        -6.75766856,
        -7.41559262,
        2.568627,
        -4.04670261,
        -15.65282253,
        8.30888474,
        -11.00462287,
        7.78068257,
        -8.39506809,
        -2.99480892,
        -2.21306447,
        -16.41295322,
        -4.95485937,
        -2.92995876,
        6.41815675,
        14.46934747,
    ]
)
