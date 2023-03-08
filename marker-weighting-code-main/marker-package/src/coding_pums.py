"""
This file defines the example coding scheme.
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
    
# map of a variable enum to the number of categorical values it requires
BIN_COUNTS = {}


#
## Age
#
class Age(Enum):
    AGE_18_29   = 0
    AGE_30_39   = 1
    AGE_40_49   = 2
    AGE_50_59   = 3
    AGE_60_69   = 4
    AGE_70_PLUS = 5

BIN_COUNTS[Variables.AGE] = len([item.value for item in Age])


#
## Race
#
class RaceEth(Enum):
    NH_WHITE = 0  # Non-Hispanic White only
    NH_BLACK = 1  # Non-Hispanic Black only 
    NH_ASIAN = 2  # Non-Hispanic Asian only
    HISPANIC = 3  # Hispanic
    OTHER    = 4  # Other

BIN_COUNTS[Variables.RACE_ETH] = len([item.value for item in RaceEth])


#
## Sex
#
class Sex(Enum):
    MALE   = 0
    FEMALE = 1

BIN_COUNTS[Variables.SEX] = len([item.value for item in Sex])


#
## Education
#
class Education(Enum):
    COLLEGE_GRAD = 0  # College grad (includes missing)
    SOME_COLLEGE = 1  # Some college
    HS_GRAD      = 2  # High school graduate
    NOT_HS_GRAD  = 3  # Not a HS graduate

BIN_COUNTS[Variables.EDUCATION] = len([item.value for item in Education])


#
## Income
#
class Income(Enum):
    INC_LT_25   = 0  # less than $25K
    INC_25_50   = 1  # $25K to $49,999
    INC_50_100  = 2  # $50K to $99,999
    INC_GT_100  = 3  # $100K or more
    
BIN_COUNTS[Variables.INCOME] = len([item.value for item in Income])


#
## Insurance
#
class Insurance(Enum):
    YES     = 0  # Has health insurance
    NO      = 1  # Does not have health insurance (includes "missing")

BIN_COUNTS[Variables.INSURANCE] = len([item.value for item in Insurance])

