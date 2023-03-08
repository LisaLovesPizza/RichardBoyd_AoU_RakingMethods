"""
U.S. Census divisions with state abbreviations and PUMS state codes.
"""

# map of state abbreviations to PUMS state codes
STATE_CODE_MAP = {
    'AL' : 1,
    'AK' : 2,
    'AZ' : 4,
    'AR' : 5,
    'CA' : 6,
    'CO' : 8,
    'CT' : 9,
    'DE' : 10,
    'DC' : 11,
    'FL' : 12,
    'GA' : 13,
    'HI' : 15,
    'ID' : 16,
    'IL' : 17,
    'IN' : 18,
    'IA' : 19,
    'KS' : 20,
    'KY' : 21,
    'LA' : 22,
    'ME' : 23,
    'MD' : 24,
    'MA' : 25,
    'MI' : 26,
    'MN' : 27,
    'MS' : 28,
    'MO' : 29,
    'MT' : 30,
    'NE' : 31,
    'NV' : 32,
    'NH' : 33,
    'NJ' : 34,
    'NM' : 35,
    'NY' : 36,
    'NC' : 37,
    'ND' : 38,
    'OH' : 39,
    'OK' : 40,
    'OR' : 41,
    'PA' : 42,
    'RI' : 44,
    'SC' : 45,
    'SD' : 46,
    'TN' : 47,
    'TX' : 48,
    'UT' : 49,
    'VT' : 50,
    'VA' : 51,
    'WA' : 53,
    'WV' : 54,
    'WI' : 55,
    'WY' : 56,
}

INV_STATE_CODE_MAP = {v:k for k,v in STATE_CODE_MAP.items()}

# map of state abbreviation to its US Census division
STATE_DIVISION_MAP = {
    'AL' : 'DIV_ESC',
    'AK' : 'DIV_P',
    'AZ' : 'DIV_M',
    'AR' : 'DIV_WSC',
    'CA' : 'DIV_P',
    'CO' : 'DIV_M',
    'CT' : 'DIV_NE',
    'DE' : 'DIV_SA',
    'DC' : 'DIV_SA',
    'FL' : 'DIV_SA',
    'GA' : 'DIV_SA',
    'HI' : 'DIV_P',
    'ID' : 'DIV_M',
    'IL' : 'DIV_ENC',
    'IN' : 'DIV_ENC',
    'IA' : 'DIV_WNC',
    'KS' : 'DIV_WNC',
    'KY' : 'DIV_ESC',
    'LA' : 'DIV_WSC',
    'ME' : 'DIV_NE',
    'MD' : 'DIV_SA',
    'MA' : 'DIV_NE',
    'MI' : 'DIV_ENC',
    'MN' : 'DIV_WNC',
    'MS' : 'DIV_ESC',
    'MO' : 'DIV_WNC',
    'MT' : 'DIV_M',
    'NE' : 'DIV_WNC',
    'NV' : 'DIV_M',
    'NH' : 'DIV_NE',
    'NJ' : 'DIV_MA',
    'NM' : 'DIV_M',
    'NY' : 'DIV_MA',
    'NC' : 'DIV_SA',
    'ND' : 'DIV_WNC',
    'OH' : 'DIV_ENC',
    'OK' : 'DIV_WSC',
    'OR' : 'DIV_P',
    'PA' : 'DIV_MA',
    'RI' : 'DIV_NE',
    'SC' : 'DIV_SA',
    'SD' : 'DIV_WNC',
    'TN' : 'DIV_ESC',
    'TX' : 'DIV_WSC',
    'UT' : 'DIV_M',
    'VT' : 'DIV_NE',
    'VA' : 'DIV_SA',
    'WA' : 'DIV_P',
    'WV' : 'DIV_SA',
    'WI' : 'DIV_ENC',
    'WY' : 'DIV_M',
}

# DIV_NE
DIV_NEW_ENGLAND = [
    ('CT', 9),   # Connecticut
    ('ME', 23),  # Maine
    ('MA', 25),  # Massachusetts
    ('NH', 33),  # New Hampshire
    ('RI', 44),  # Rhode Island
    ('VT', 50),  # Vermont
]

# DIV_MA
DIV_MIDDLE_ATLANTIC = [
    ('NJ', 34), # New Jersey
    ('NY', 36), # New York
    ('PA', 42), # Pennsylvania
]

# DIV_ENC
DIV_EAST_NORTH_CENTRAL = [
    ('IN', 18), # Indiana
    ('IL', 17), # Illinois
    ('MI', 26), # Michigan
    ('OH', 39), # Ohio
    ('WI', 55), # Wisconsin
]

# DIV_WNC
DIV_WEST_NORTH_CENTRAL = [
    ('IA', 19), # Iowa
    ('KS', 20), # Kansas
    ('MN', 27), # Minnesota
    ('MO', 29), # Missouri
    ('NE', 31), # Nebraska
    ('ND', 38), # North Dakota
    ('SD', 46), # South Dakota
]

# DIV_SA
DIV_SOUTH_ATLANTIC = [
    ('DE', 10), # Delaware
    ('DC', 11), # District of Columbia
    ('FL', 12), # Florida
    ('GA', 13), # Georgia
    ('MD', 24), # Maryland
    ('NC', 37), # North Carolina
    ('SC', 45), # South Carolina
    ('VA', 51), # Virginia
    ('WV', 54), # West Virginia
]

# DIV_ESC
DIV_EAST_SOUTH_CENTRAL = [
    ('AL', 1),  # Alabama
    ('KY', 21), # Kentucky
    ('MS', 28), # Mississippi
    ('TN', 47), # Tennessee
]

# DIV_WSC
DIV_WEST_SOUTH_CENTRAL = [
    ('AR', 5),  # Arkansas
    ('LA', 22), # Louisiana
    ('OK', 40), # Oklahoma
    ('TX', 48), # Texas
]

# DIV_M
DIV_MOUNTAIN = [
    ('AZ', 4),  # Arizona
    ('CO', 8),  # Colorado
    ('ID', 16), # Idaho
    ('NM', 35), # New Mexico
    ('MT', 30), # Montana
    ('UT', 49), # Utah
    ('NV', 32), # Nevada
    ('WY', 56), # Wyoming
]

# DIV_P
DIV_PACIFIC = [
    ('AK', 2),  # Alaska
    ('CA', 6),  # California
    ('HI', 15), # Hawaii
    ('OR', 41), # Oregon
    ('WA', 53), # Washington
]

# map an abbreviation to list of states and PUMS codes
CENSUS_DIV_MAP = {
    'DIV_NE'  : DIV_NEW_ENGLAND,
    'DIV_MA'  : DIV_MIDDLE_ATLANTIC,
    'DIV_ENC' : DIV_EAST_NORTH_CENTRAL,
    'DIV_WNC' : DIV_WEST_NORTH_CENTRAL,
    'DIV_SA'  : DIV_SOUTH_ATLANTIC,
    'DIV_ESC' : DIV_EAST_SOUTH_CENTRAL,
    'DIV_WSC' : DIV_WEST_SOUTH_CENTRAL,
    'DIV_M'   : DIV_MOUNTAIN,
    'DIV_P'   : DIV_PACIFIC
}


def is_valid(source_state, target_state):
    """
    Validate the source and target states. The arguments are:

    source_state: A two-letter state abbreviation or a census division key
                  from the CENSUS_DIV_MAP above.

    target_state: A two-letter state abbreviation, cannot be a census division.
                  If the source_state is a census division, the target_state
                  must exist in that same division.
    """

    # validate the source state
    if not source_state in STATE_CODE_MAP and not source_state in CENSUS_DIV_MAP:
        print('Error: census_divisions.is_valid: invalid source state code "{0}"'.format(source_state))
        return False

    # validate the TARGET_STATE
    if not target_state in STATE_CODE_MAP:
        print('Error: census_divisions.is_valid: invalid target state code "{0}"'.format(target_state))
        return False
    
    # target state must be in census division
    if source_state in CENSUS_DIV_MAP:
        if source_state != STATE_DIVISION_MAP[target_state]:
            print('Error: census_divisions.is_valid: target state "{0}" not a member of census division "{1}"'.
                  format(target_state, source_state))
            return False

    return True
