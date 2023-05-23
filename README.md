### Debye Temperature ML
### Author: Adam D. Smith 
### Institution: Department of Physics, University of Alabama at Birmingham

### Paper now available as a preprint: https://arxiv.org/pdf/2305.12977.pdf
___________________________________________________________________________
#### Dependencies:  Python>=3.7, Python packages: Matminer, Pandas, XGboost
___________________________________________________________________________
### Predict the Debye Temperature of many compounds in 2 easy steps:

#### 1. Edit the Test.csv file with Excel or similar spreadsheet software with the compounds you want to predict and their lattice type. 

#### 2. Run the Debye_ML.py script with: 
python3 Debye_ML.py
#### This will return a Test_out.csv file with machine learned predictions for the Debye temperature of the compounds specified in the Test.csv file.
