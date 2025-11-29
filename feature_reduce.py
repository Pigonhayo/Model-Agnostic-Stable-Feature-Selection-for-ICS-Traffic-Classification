import pandas as pd

# CSV 불러오기
df = pd.read_csv("/home/ice06/project/secure/mrmr_test/advice/Dataset.csv")

# 선택할 23개 feature
selected_features = [
    "end",
    "rBytesSum",
    "endOffset",
    "start",
    "sBytesSum",
   "startOffset",
   "rPayloadSum",
   "sPayloadSum",
   "rPackets",
   "sPackets",
   "rBytesAvg",
   "rInterPacketAvg",
   "sLoad",
   "rLoad",
   "sInterPacketAvg",
   "rPayloadAvg",
   "rAckDelayAvg",
   "rPshRate",
   "sAckDelayAvg",
   "sBytesAvg",
   "rAckDelayMax",
   "sPayloadAvg",
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
df_reduced.to_csv("/home/ice06/project/secure/mrmr_test/advice/dataset/cv_baseline/cv_baseline.csv", index=False)
