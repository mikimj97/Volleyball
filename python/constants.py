# Sampling rate in # per second
SAMPLING_RATE = 13

JUMP_TYPES = {
    "prep": {
        "none": 0,
        "BASIC": 1,
        "Q3L": 2,
        "Q3R": 3,
        "X3L": 4,
        "X3R": 5,
        "X2L": 6,
        "X2R": 7,
        "ATTACK": 8
    },
    "run": {
        "BASIC": 0,
        "Q3L": 1,
        "Q3R": 2,
        "X3L": 3,
        "X3R": 4,
        "X2L": 5,
        "X2R": 6,
        "ATTACK": 7
    }
}

JUMP_INDEX_TO_TYPE = {
    1: "BASIC",
    2: "Q3L",
    3: "Q3R",
    4: "X3L",
    5: "X3R",
    6: "X2L",
    7: "X2R",
    8: "ATTACK"
}

INITIALS = {
    "EA": 0,
    "GJ": 1,
    "IL": 2,
    "JP": 3,
    "KE": 4,
    "MJ": 5,
    "MR": 6,
    "JT": 7, #TJT
    "TN": 8,
    "TT": 9,
    "WB": 10
}