"""

This file specifies a coding scheme that closely matches PUMS.

Variables: Age, RaceEth, Sex, Education, Income, Insurance

Differences from the PUMS coding scheme:

    The age and income variables are now categorical.

    Race and ethnicity have been combined into a single RaceEth variable.

    A MISSING category has been added to the Education field, since nearly 2.7%
    of the Education values are missing.

    A MISSING category has been added to the Income field, since nearly 15.5%
    of the Income values are missing.

"""

from enum import Enum

# list of variables enumerated in this file
class Variables(Enum):
    AGE       = 0
    RACE_ETH  = 1  # (combined race and ethnicity)
    SEX       = 2
    EDUCATION = 3
    INCOME    = 4
    INSURANCE = 5

# map of variable enum ID to name string
VAR_NAMES = {
    Variables.AGE       : 'Age',
    Variables.RACE_ETH  : 'RaceEth',
    Variables.SEX       : 'Sex',
    Variables.EDUCATION : 'Education',
    Variables.INCOME    : 'Income',
    Variables.INSURANCE : 'Insurance',
}
    
# map of variable names to enum values, use as array indices
NAMES_TO_INDICES = {
    VAR_NAMES[Variables.AGE]       : Variables.AGE.value,
    VAR_NAMES[Variables.RACE_ETH]  : Variables.RACE_ETH.value,
    VAR_NAMES[Variables.SEX]       : Variables.SEX.value,
    VAR_NAMES[Variables.EDUCATION] : Variables.EDUCATION.value,
    VAR_NAMES[Variables.INCOME]    : Variables.INCOME.value,
    VAR_NAMES[Variables.INSURANCE] : Variables.INSURANCE.value,
}

INDICES_TO_NAMES = {v:k for k,v in NAMES_TO_INDICES.items()}

# total number of variables
TOTAL_VARS = len([item.value for item in Variables])

# map of a variable name to the number of categorical values it requires
BIN_COUNTS = {}


###############################################################################
# AGE (PUMS variable AGEP, rebinned from single-year bins)
###############################################################################

class Age(Enum):
    AGE_0_17    = 0  #  0-17 years old
    AGE_18_24   = 1  # 18-24 years old
    AGE_25_34   = 2  # 25-34 years old
    AGE_35_44   = 3  # 35-44 years old
    AGE_45_54   = 4  # 45-54 years old
    AGE_55_64   = 5  # 55-64 years old
    AGE_65_PLUS = 6  # 65 or older

BIN_COUNTS[Variables.AGE] = len([item.value for item in Age])


###############################################################################
# RACE_ETH (combination of PUMS variables RAC1P and HISP)
###############################################################################

class RaceEth(Enum):
    WHITE     = 0  # White alone
    BLACK     = 1  # Black or African American alone
    ASIAN     = 2  # Asian alone
    OTHER     = 3  # Other race, multiracial
    HISPANIC  = 4  # Hispanic

    
BIN_COUNTS[Variables.RACE_ETH] = len([item.value for item in RaceEth])


###############################################################################
# SEX (PUMS variable SEX)
###############################################################################

class Sex(Enum):
    MALE   = 0
    FEMALE = 1

# map of PUMS SEX value to recoded value
SEX_RECODE_MAP = {
    1 : Sex.MALE.value,
    2 : Sex.FEMALE.value,
}

BIN_COUNTS[Variables.SEX] = len([item.value for item in Sex])


###############################################################################
# EDUCATION (PUMS variable SCHL)
#
# The NA value does not occur in the PUMS 2020 data.
###############################################################################

# class Education(Enum):
#     PRE               = 0  # Never attended, preschool, kindergarten
#     G1                = 1  # Grade 1
#     G2                = 2  # Grade 2
#     G3                = 3  # Grade 3
#     G4                = 4
#     G5                = 5
#     G6                = 6
#     G7                = 7
#     G8                = 8
#     G9                = 9
#     G10               = 10
#     G11               = 11
#     G12_ND            = 12  # 12th grade - no diploma
#     HS_DIPLOMA        = 13  # Regular high school diploma
#     GED               = 14  # GED or alternative credential
#     SOME_COLLEGE      = 15  # Combine the next two
#     DEG_ASSOC         = 16  # Associate's degree
#     DEG_BACHELORS     = 17  # Bachelor's degree
#     DEG_MASTERS       = 18  # Master's degree
#     DEG_PROF          = 19  # Professional degree beyond a bathelor's degree
#     DEG_PHD           = 20  # Doctorate degree
#     MISSING           = 21  # approx. 2.7% of the PUMS SCHL fields are missing

class Education(Enum):
    LT_HS           = 0  # None, preschool, kindergarten, grades 1-12
    HS_GRAD         = 1  # Regular high school diploma or GED
    SOME_COLLEGE    = 2  # Combine the next two
    DEG_ASSOC       = 3  # Associate's degree
    DEG_BACHELORS   = 4  # Bachelor's degree
    DEG_GRADUATE    = 5  # Master's, Ph.D., prof degree
    #MISSING         = 6  # approx. 2.7% of the PUMS SCHL fields are missing

    
BIN_COUNTS[Variables.EDUCATION] = len([item.value for item in Education])


###############################################################################
# INCOME (PUMS variable PINCP, rebinned)
#
# The NA and INC_LOSS_2 values do not occur in the PUMS 2020 data.
###############################################################################

class Income(Enum):
    INC_LT_25   = 0  # Less than $25K
    INC_25_50   = 1  # $25K to $49,999
    INC_50_100  = 2  # $50K to $99,999
    INC_GT_100  = 3  # $100K or more
    INC_MISSING = 4  # Missing 
    
BIN_COUNTS[Variables.INCOME] = len([item.value for item in Income])


###############################################################################
# INSURANCE (PUMS variable HICOV)
###############################################################################

class Insurance(Enum):
    YES = 0
    NO  = 1

INSURANCE_RECODE_MAP = {
    1 : Insurance.YES.value,
    2 : Insurance.NO.value,
}

BIN_COUNTS[Variables.INSURANCE] = len([item.value for item in Insurance])
