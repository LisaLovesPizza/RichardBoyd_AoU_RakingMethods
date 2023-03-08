"""

This file specifies the coding scheme for raking of survey data to
match census totals.


Variables: Race, Education, Insurance, Income, Sex

"""

VARIABLES = [
    'Race',       # 0
    'Education',  # 1
    'Insurance',  # 2
    'Income',     # 3
    'Sex'         # 4
]

VAR_INDICES = {v:VARIABLES.index(v) for v in VARIABLES}

# map of a variable name to the number of categorical values it requires
BIN_COUNTS = {}


###############################################################################
# R A C E
###############################################################################

# 0 : Non-Hispanic White Only
# 1 : Non-Hispanic Black Only
# 2 : Non-Hispanic Asian Only
# 3 : Hispanic
# 4 : Other

RACE_NH_WHITE = 0
RACE_NH_BLACK = 1
RACE_NH_ASIAN = 2
RACE_HISPANIC = 3
RACE_OTHER    = 4

RACE_CODES = [
    RACE_NH_WHITE, RACE_NH_BLACK, RACE_NH_ASIAN, RACE_HISPANIC, RACE_OTHER
]

BIN_COUNTS['Race'] = len(RACE_CODES)


###############################################################################
# E D U C A T I O N
###############################################################################

# 0 : College graduate
# 1 : Some college
# 2 : High school graduate
# 3 : Not a HS graduate
# 4 : Missing    

EDUC_COLLEGE_GRAD = 0
EDUC_SOME_COLLEGE = 1
EDUC_HS_GRAD      = 2
EDUC_NOT_HS_GRAD  = 3
EDUC_MISSING      = 4

EDUC_CODES = [
    EDUC_COLLEGE_GRAD, EDUC_SOME_COLLEGE, EDUC_HS_GRAD,
    EDUC_NOT_HS_GRAD, EDUC_MISSING
]

BIN_COUNTS['Education'] = len(EDUC_CODES)


###############################################################################
# I N S U R A N C E
###############################################################################

# 0 : Has health insurance
# 1 : Does not have health insurance
# 2 : Missing

INS_YES     = 0
INS_NO      = 1
INS_MISSING = 2

INS_CODES = [
    INS_YES, INS_NO, INS_MISSING
]

BIN_COUNTS['Insurance'] = len(INS_CODES)


###############################################################################
# I N C O M E
###############################################################################

# 0 : less than $25K
# 1 : [$25K, $50K)
# 2 : [$50K, 100K)
# 3 : $100K or greater
# 4 : Missing

INCOME_LT_25   = 0
INCOME_25_50   = 1
INCOME_50_100  = 2
INCOME_GT_100  = 3
INCOME_MISSING = 4

INCOME_CODES = [
    INCOME_LT_25, INCOME_25_50, INCOME_50_100, INCOME_GT_100, INCOME_MISSING
]

BIN_COUNTS['Income'] = len(INCOME_CODES)


###############################################################################
# S E X
###############################################################################

# 0 : Male
# 1 : Female
# 2 : Other

SEX_MALE   = 0
SEX_FEMALE = 1
SEX_OTHER  = 2

SEX_CODES = [
    SEX_MALE, SEX_FEMALE, SEX_OTHER
]

BIN_COUNTS['Sex'] = len(SEX_CODES)

