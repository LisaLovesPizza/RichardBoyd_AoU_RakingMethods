"""
U.S. Census divisions with state abbreviations and PUMS state codes.

States with 1000 or more samples in AllOfUs are denoted with an asterisk.
"""

# DIV_NE
DIV_NEW_ENGLAND = [
    ('CT', 9),   # Connecticut*
    ('ME', 23),  # Maine
    ('MA', 25),  # Massachusetts*
    ('NH', 33),  # New Hampshire
    ('RI', 44),  # Rhode Island
    ('VT', 50),  # Vermont
]

# DIV_MA
DIV_MIDDLE_ATLANTIC = [
    ('NJ', 34), # New Jersey
    ('NY', 36), # New York*
    ('PA', 42), # Pennsylvania*
]

# DIV_ENC
DIV_EAST_NORTH_CENTRAL = [
    ('IN', 18), # Indiana
    ('IL', 17), # Illinois*
    ('MI', 26), # Michigan*
    ('OH', 39), # Ohio
    ('WI', 55), # Wisconsin*
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
    ('FL', 12), # Florida*
    ('GA', 13), # Georgia*
    ('MD', 24), # Maryland
    ('NC', 37), # North Carolina
    ('SC', 45), # South Carolina*
    ('VA', 51), # Virginia
    ('WV', 54), # West Virginia
]

# DIV_ESC
DIV_EAST_SOUTH_CENTRAL = [
    ('AL', 1),  # Alabama*
    ('KY', 21), # Kentucky
    ('MS', 28), # Mississippi*
    ('TN', 47), # Tennessee*
]

# DIV_WSC
DIV_WEST_SOUTH_CENTRAL = [
    ('AR', 5),  # Arkansas
    ('LA', 22), # Louisiana*
    ('OK', 40), # Oklahoma
    ('TX', 48), # Texas*
]

# DIV_M
DIV_MOUNTAIN = [
    ('AZ', 4),  # Arizona*
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
    ('CA', 6),  # California*
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

# These divisions need custom collapsing per memo 5.4.
# Add two-letter abbreviations and PUMS codes to each set.
STATES_DIV_WNC = set()
STATES_DIV_ESC = set()
STATES_DIV_WSC = set()
for abbrev, code in DIV_WEST_NORTH_CENTRAL:
    STATES_DIV_WNC.add(abbrev)
    STATES_DIV_WNC.add(code)
for abbrev, code in DIV_EAST_SOUTH_CENTRAL:
    STATES_DIV_ESC.add(abbrev)
    STATES_DIV_ESC.add(code)
for abbrev, code in DIV_WEST_SOUTH_CENTRAL:
    STATES_DIV_WSC.add(abbrev)
    STATES_DIV_WSC.add(code)
