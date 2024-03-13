# SD County Trip Destination Prediction Model

## Some Notes:
### Input data is already located in the repository.
The original data was provided by the San Diego Association of Governments synthetic population and census data. The relevant files have been uploaded to the repository already, so there is no need to get it yourself.
### All model visualizations and outputs will be in the "data/out" folder after running the models.

## Steps to Run Code:
### 1. Create directory for this project to be stored locally.
```
mkdir sd_trip_distance_pred
```
```
cd sd_trip_distance_pred
```

### 2. Clone the repository from Github.
The following commands will clone the repository from Github onto your local machine.
```
git clone -b q2_project https://github.com/ryansliao/SD-County-Trip-Destination-Prediction.git
```
```
cd SD-County-Trip-Destination-Prediction
```

### 3. Download raw data files from Google Drive.
Download the "raw" folder and the "final_trips.csv" file at the link below.
When you do this, you will get a zipped folder plus the final_trips.csv. Unzip the folder, and place the final_trips.csv into the unzipped folder.
Then, place that folder into the "data" folder in your cloned repository.
URL: https://drive.google.com/drive/folders/1c1fkr4h4avE5toLwMwMsUjqAJGw-68Dw?usp=drive_link

### 4. Create a virtual environment locally.
The following commands create local virtual environment.
```
pip install virtualenv
```
```
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

### 5. Install environment requirements.
The following commands installs all of the required libraries and packages for this project.
```
pip install -r requirements.txt
```

### 6. Run code.
To get data features:
```
python run.py data features
```

To run the model:
```
python run.py data features model
```

### 7. Deactivate virtual environment.
The following command deactivates the virtual environment once you are done running the code.
```
deactivate
```
