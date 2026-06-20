# Data analysis

## String columns:

large variaty of names/surnames
name and surname to house shows no correlation, safe to drop

## Enum

Best hand should be converted Left/Right to 0/1

## Correlations

Building correlation matrix for all the features: results
 - correlation between DADA and Astonomy is -1
 - strong correlation between History and Flying 0.9
 - strong correlation between Transfigration and Flying, 0.87

Suggestions: 
 - get rid of Flying and DADA

## N/A Options:

1) dropna: best if data is missing and not too much info is lost
2) replace with avg: best if data if significant part of data will be lost across the columns
3) add feature if present or not: best if missing data has some information, e.g. class was not taken
 - add 1 bitfield feature
 - add 1 bool feature per original feature 
4) add out of distribution value (e.g if val is distributed 0..1 replace with -1)
5) ...

