"""
This file defines the coding scheme, i.e. all variables and
all categorical values for each.
"""

from enum import Enum


# list of variables enumerated in this file
class Variables(Enum):
    URBRUR     = 0  # Urban or Rural
    TEACHING   = 1  # Whether teaching hospital or not
    BEDS       = 2  # Number of beds
    REGION     = 3  # US Census region
    OWNERSHIP  = 4  # Government, profit, non-profit

# map of variable enum ID to name string
VAR_NAMES = {
    Variables.URBRUR     : 'UrbRur',
    Variables.TEACHING   : 'Teaching',
    Variables.BEDS       : 'TotBeds',
    Variables.REGION     : 'Region',
    Variables.OWNERSHIP  : 'Ownership',
}
    
# map of a variable enum to the number of categorical values it requires
BIN_COUNTS = {}


#
## URBRUR
#
class UrbRur(Enum):
    URBAN = 0
    RURAL = 1

BIN_COUNTS[Variables.URBRUR] = len([item.value for item in UrbRur])


#
## TEACHING
#
class Teaching(Enum):
    NO  = 0
    YES = 1

BIN_COUNTS[Variables.TEACHING] = len([item.value for item in Teaching])


#
## BEDS
#
class Beds(Enum):
    BEDS_006_099  = 0
    BEDS_100_199  = 1
    BEDS_200_299  = 2
    BEDS_300_499  = 3
    BEDS_500_PLUS = 4

BIN_COUNTS[Variables.BEDS] = len([item.value for item in Beds])


#
## REGION
#
class Region(Enum):
    MIDWEST   = 0
    SOUTH     = 1
    NORTHEAST = 2
    WEST      = 3

BIN_COUNTS[Variables.REGION] = len([item.value for item in Region])


#
## OWNERSHIP
#
class Ownership(Enum):
    GOVERNMENT = 0
    PROFIT     = 1
    NONPROFIT  = 2
    
BIN_COUNTS[Variables.OWNERSHIP] = len([item.value for item in Ownership])