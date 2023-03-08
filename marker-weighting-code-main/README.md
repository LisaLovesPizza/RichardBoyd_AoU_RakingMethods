# t10-raking
Rake some survey data to match Census demographics using the variables age, race, education, insurance, income, and sex.

`PUMS-Preprocessing.ipynb` : downloads all PUMS person-level files for a given year, extracts required columns, concatenates in memory, and writes out to disk.

`PUMS-Recoding.ipynb` : loads the output file from `PUMS-Preprocessing` and recodes it according to the coding scheme in `src/coding_pums.py`.

`NHIS-Recoding.ipynb` : loads some data from the 2018 National Health Interview Survey and recodes it according to the coding scheme in `src/coding_pums.py`.

`Raking-General.ipynb` : loads recoded source and target files and rakes the source to match specified target marginals.

