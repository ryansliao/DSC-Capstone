# DSC-Capstone

## Some Notes:
### Input data is already located in the repository.
The original data comes from the National Household Travel Survey website. It has been uploaded to the repository already, so there is no need to get it yourself.
### All model visualizations and outputs will be in an "outputs" folder after running the models.

## Steps to Run Code:
### 1. Clone the repository from Github.
The following command will clone the repository from Github onto your local machine.
```
git clone https://github.com/ryansliao/DSC-Capstone.git
```

### 2. Create a virtual environment locally.
The following commands create local virtual environment.
```
python -m venv myenv
myenv\Scripts\activate
```

### 3. Install environment requirements.
The following command installs all of the required libraries and packages for this project.
```
pip install -r requirements.txt
```

### 4. Run code
Our larger vehicle type choice model contains three separate models that you can run all at once, or individually.

To run all at once:
```
python run.py data features all
```

To run individually:
```
python run.py data features vehtype
```
```
python run.py data features fueltype
```
```
python run.py data features vehage
```
