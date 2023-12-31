# DSC-Capstone

## Some Notes:
### Input data is already located in the repository.
The original data comes from the National Household Travel Survey website. It has been uploaded to the repository already, so there is no need to get it yourself.
### All model visualizations and outputs will be in an "outputs" folder after running the models.

## Steps to Run Code:
### 1. Create directory for this project to be stored locally.
```
mkdir capstone_project
```

### 2. Clone the repository from Github.
The following commands will clone the repository from Github onto your local machine.
```
git clone https://github.com/ryansliao/DSC-Capstone.git
cd DSC-Capstone
```

### 3. Create a virtual environment locally.
The following commands create local virtual environment.
```
pip install virtualenv
python -m venv myenv
```

For Windows:
```
myenv\Scripts\activate
```

For Mac/Linux:
```
source myenv/bin/activate
```

### 4. Install environment requirements.
The following commands installs all of the required libraries and packages for this project.
```
pip install -r requirements.txt
```

### 5. Run code.
Our larger vehicle type choice model contains three separate models that you can run all at once, or individually. You can also choose to not run any models and just get the data features.

To get data features:
```
python run.py data features
```

To run all models at once:
```
python run.py data features all
```

To run each model individually:
```
python run.py data features vehtype
```
```
python run.py data features fueltype
```
```
python run.py data features vehage
```

### 6. Deactivate virtual environment.
The following command deactivates the virtual environment once you are done running the code.
```
deactivate
```
