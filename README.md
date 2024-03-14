# SD County Trip Destination Prediction Model

We have collaborated with the San Diego Association of Governments (SANDAG) to improve upon their existing activity based model (ABM). An ABM is a series of models which utilizes a synthetically generated population and produces daily travel habits for that population. The idea is to run this series of models to get a better understanding of the population and how it interacts with the area they live in. We can then use these insights to make better public policy decisions. However, many of these models are slow and inefficient. ABMs predominantly features statistical models that take a long time to process millions of trips and people to produce a final result. Our goal was to introduce machine learning methodology to these models, which would hopefully increase time efficiency while not sacrificing performance and accuracy. We were able to accomplish this goal by using a decision tree classifier, SMOTE to address data class imbalance, and a grid search how tune our parameters to optimize our model's final performance. This method increased time efficiency by ~9x and produced an almost identical trip distance distribution as SANDAG's model, while maintaining an approximate F1 score of 73%. We deem these final metrics as satisfactory considering our original goal.

The original data was provided by the San Diego Association of Governments synthetic population and census data. The relevant files have to be downloaded from the Google Drive link below.
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
