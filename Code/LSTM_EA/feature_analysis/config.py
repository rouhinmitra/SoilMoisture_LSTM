"""
Configuration for moisture proxy / feature diagnostic analysis.
Single source of truth for column names and year range.
"""

STATION_FILES = {
    'Ne1': 'ne1_1_maize.csv',
    'Ne2': 'ne2_1_maize.csv',
    'Ne3': 'ne3_1_maize.csv',
}

TARGET_COL = 'RZSM_25_avg'

SSM_COLS = ['SSM', 'SSM_avg', 'SWC_PI_F_2_1_1', 'SWC_PI_F_3_1_1']

S2_COLS = ['ndvi', 'b11', 'b12', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b8a']

METEO_PRECIP_COLS = ['TA_1_1_1', 'RH_1_1_1', 'P_PI_F_1_1_1', 'P_PI_F_2_2_1', 'I']

AE_COLS = [f'A{i:02d}' for i in range(64)]

PRESTO_COLS = [f'emb_{k}' for k in range(128)]

YEAR_RANGE = (2017, 2024)
