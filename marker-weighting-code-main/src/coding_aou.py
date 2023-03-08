"""
This file defines the AllOfUs coding scheme from Memo 5.4.

I added a MISSING category for Income since AOU has a substantial number of
individuals in this category. This will be useful for runs that do *not*
use the imputed income file.
"""

from enum import Enum
from . import census_divisions as census


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

# map of a variable enum to the number of categorical values it requires
BIN_COUNTS = {}


###############################################################################
# AGE
###############################################################################

class Age(Enum):
    AGE_18_29   = 0
    AGE_30_39   = 1
    AGE_40_49   = 2
    AGE_50_59   = 3
    AGE_60_69   = 4
    AGE_70_PLUS = 5

BIN_COUNTS[Variables.AGE] = len([item.value for item in Age])


###############################################################################
# R A C E
###############################################################################

class RaceEth(Enum):
    NH_WHITE = 0  # Non-Hispanic White only
    NH_BLACK = 1  # Non-Hispanic Black only 
    NH_ASIAN = 2  # Non-Hispanic Asian only
    HISPANIC = 3  # Hispanic
    OTHER    = 4  # Other

BIN_COUNTS[Variables.RACE_ETH] = len([item.value for item in RaceEth])


###############################################################################
# S E X
###############################################################################

class Sex(Enum):
    MALE   = 0 # includes "Other"
    FEMALE = 1

BIN_COUNTS[Variables.SEX] = len([item.value for item in Sex])


###############################################################################
# E D U C A T I O N
###############################################################################

class Education(Enum):
    COLLEGE_GRAD = 0  # College grad (includes missing)
    SOME_COLLEGE = 1  # Some college
    HS_GRAD      = 2  # High school graduate
    NOT_HS_GRAD  = 3  # Not a HS graduate

BIN_COUNTS[Variables.EDUCATION] = len([item.value for item in Education])


###############################################################################
# I N C O M E
###############################################################################

class Income(Enum):
    INC_LT_25   = 0  # less than $25K
    INC_25_50   = 1  # $25K to $49,999
    INC_50_100  = 2  # $50K to $99,999
    INC_GT_100  = 3  # $100K or more
    INC_MISSING = 4  # Missing 
    
BIN_COUNTS[Variables.INCOME] = len([item.value for item in Income])


###############################################################################
# I N S U R A N C E
###############################################################################

class Insurance(Enum):
    YES     = 0  # Has health insurance
    NO      = 1  # Does not have health insurance (includes "missing")

BIN_COUNTS[Variables.INSURANCE] = len([item.value for item in Insurance])


###############################################################################
#
#             S T A T E - S P E C I F I C  C O L L A P S I N G
#
###############################################################################
def collapse_age(df, state_col, age_col):
    """
    State-specific collapsing of AGE according to memo 5.4.
    """

    new_values = []
    for index, row in df.iterrows():
        # for CT: collapse 18-29 into 30-39 (i.e. change Age==0 to Age==1)
        # for SC: collapse 70+ into 60-69 (i.e. change Age==5 to Age==4)
        state = row[state_col]
        age = row[age_col]
        assert age >=0 and age <= 5
        if ('CT' == state or 9 == state) and 0 == age:
            new_age = 1
        elif ('SC' == state or 45 == state) and 5 == age:
            new_age = 4
        # elif state in census.STATES_DIV_WNC:
        #     # West North Central census division, rebin as follows:
        #     # bin 0: 18-49  (Age==0, 1, and 2 => 0)
        #     # bin 1: 50-69  (Age==3, 4 => 1)
        #     # bin 2: 70+    (Age==5 => 2)
        #     if age <= 2:
        #         new_age = 0
        #     elif age <= 4:
        #         new_age = 1
        #     else:
        #         new_age = 2
        elif state in census.STATES_DIV_WNC:
            # West North Central census division, rebin as follows:
            # bin 2: 18-49  (change Age==0 and Age==1 to Age==2)
            # bin 3: 50-69  (change Age==3 to Age==4)
            # leave bin 5 (70+ as is)
            if age <= 1:
                new_age = 2
            elif 3 == age:
                new_age = 4
            else:
                new_age = age
        else:
            new_age = age
        new_values.append(new_age)
        
    df = df.assign(**{age_col:new_values})    
    return df


###############################################################################
def collapse_raceeth(df, state_col, race_col):
    """
    State-specific collapsing of RACE_ETH according to memo 5.4.
    """

    new_values = []
    for index, row in df.iterrows():
        # for CT, TN, East South Central and West South Central:
        #     collapse Asian into Other (i.e. change RaceEth==2 to RaceEth==4)
        # for SC: collapse Asian and Other into White (i.e. change RaceEth==2 or 4 to 0)
        # for LA and MS: collapse Asian and Hispanic into Other (i.e. change RaceEth==2,3 to 4)
        # for West North Central states: collapse Black and Asian into Other (i.e. change RaceEth==1,2 to 4)
        state = row[state_col]
        race_eth = row[race_col]
        assert race_eth >= 0 and race_eth <= 4
        if 2 == race_eth and (state in {'CT', 9, 'TN', 47} or state in census.STATES_DIV_ESC or \
            state in census.STATES_DIV_WSC):
            new_re = 4
        elif ('SC' == state or 45 == state) and race_eth in {2, 4}:
            new_re = 0
        elif state in {'LA', 22, 'MS', 28} and race_eth in {2, 3}:
            new_re = 4
        elif state in census.STATES_DIV_WNC and race_eth in {1, 2}:
            new_re = 4
        else:
            new_re = race_eth
        new_values.append(new_re)

    df = df.assign(**{race_col:new_values})
    return df


###############################################################################
def collapse_education(df, state_col, edu_col):
    """
    State-specific collapsing of EDUCATION according to memo 5.4.
    """

    new_values = []
    for index, row in df.iterrows():
        # for West North Central states: collapse NOT_HS_GRAD with HS_GRAD (i.e. change code 3 to code 2)
        state = row[state_col]
        edu = row[edu_col]
        assert edu >= 0 and edu <= 3
        if state in census.STATES_DIV_WNC and 3 == edu:
            new_edu = 2
        else:
            new_edu = edu
        new_values.append(new_edu)

    df = df.assign(**{edu_col:new_values})
    return df


###############################################################################
def collapse_income(df, state_col, inc_col):
    """
    State-specific collapsing of INCOME according to memo 5.4.
    """

    new_values = []
    for index, row in df.iterrows():
        # for CT, MS, TN, and SC: collapse > $100K into $50-$100K (i.e. change code 3 to code 2)
        # for West North Central states: collapse $25-$50K into <$25K (i.e. change code 1 to code 0)
        state = row[state_col]
        income = row[inc_col]
        assert income >= 0 and income <=4
        if 3 == income and state in {'CT', 9, 'MS', 28, 'TN', 47, 'SC', 45}:
            new_income = 2
        elif 1 == income and state in census.STATES_DIV_WNC:
            new_income = 0
        else:
            new_income = income
        new_values.append(new_income)
        
    df = df.assign(**{inc_col:new_values})
    return df
