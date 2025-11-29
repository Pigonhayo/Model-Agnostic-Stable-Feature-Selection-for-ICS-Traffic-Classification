import pandas as pd

# CSV 불러오기
df = pd.read_csv("/home/ice06/project/secure/mrmr_test/Dataset.csv")

# 선택할 23개 feature
selected_features = [
    "endOffset",
    "startOffset",
    "end",
    "start",
    "rBytesSum",
    "sBytesSum",
    "sPayloadSum",
    "rPayloadSum",
    "rPackets",
    "rInterPacketAvg",
    "rLoad",
    "sLoad",
    "sPackets",
    "rBytesAvg",
    "sInterPacketAvg",
    "rPayloadAvg",
    "rAckDelayAvg",
    "sAckDelayAvg",
    "rPshRate",
    "rAckDelayMax",
    "sBytesAvg",
    "duration",
    "sPayloadAvg",
    "sWinTCP",
    "rWinTCP",
    "rBytesMin",
    "sPshRate",
    "sBytesMin",
    "sAckDelayMax",
    "rBytesMax",
    "NST_M_Label"
]

""" 주석처리
    "sBytesAvg", "rBytesAvg",
    "sFinRate",
    #"rFinRate",
    #"sSynRate",
    #"rSynRate",
    #"sRstRate", "rRstRate",
    "sttl", "rttl",
    #"sAckRate",
    #"rAckRate",
    "sAckDelayMax", "rAckDelayMax",
    "sPackets", "rPackets",
    "protocol",
    "sWinTCP", "rWinTCP",
    #"sPayloadAvg", "rPayloadAvg",
    "sInterPacketAvg", "rInterPacketAvg",
    "NST_M_Label"
"""

# 이 23개 컬럼만 남기기
df_reduced = df[selected_features]

# 결과 저장
df_reduced.to_csv("/home/ice06/project/secure/mrmr_test/dataset/check_new/check_new.csv", index=False)
