import matplotlib.pyplot as plt  # Importing matplotlib for visualization
import pandas as pd  # Importing pandas for data manipulation and analysis
import numpy as np  # Importing numpy for numerical operations
import xlrd  # Importing xlrd to read data from Excel files
import xlwt  # Importing xlwt to write data to Excel files
import joblib  # Importing joblib to save and load models
import pyautogui  # Importing pyautogui for GUI automation
import math  # Importing math for mathematical operations
import pyperclip  # Importing pyperclip for clipboard operations
import cv2  # Importing cv2 for image processing
import csv  # Importing csv for reading and writing CSV files
import random  # Importing random for random number generation
import warnings  # Importing warnings to ignore warnings
import os  # Importing os for operating system-related operations
import time  # Importing time to measure and manipulate time
from eppy.modeleditor import IDF  # Importing IDF from eppy to edit and manipulate EnergyPlus models
from eppy.results import readhtml  # Importing readhtml from eppy.results to read EnergyPlus HTML output files
from ml_utils import *  # Importing ml_utils to use machine learning utilities


# Ignore warnings
warnings.filterwarnings("ignore")

# Set working directory
os.chdir('E:/surrogate model')

# Set EnergyPlus IDD file
iddfile = "Energy+.idd"


# Define the simulated sleep times in hours
Simulate_sleep_times = [0, 3, 6, 10, 14, 18, 22, 26, 28, 32, 36]

# Define the different envelope paths
Envelope_Paths = ['reference', 'moderate', 'radical']

# Define the different weather scenarios
Weather_Scenarios = ['ssp126', 'ssp245', 'ssp585']

# Define the years to observe
year_observes = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043,
                 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067,
                 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2088, 2089, 2090, 2091,
                 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2100]

# List of variables used for heating model training
Model_training_variable_heating = ['roof_u-factor', 'wall_u-factor', 'window_u-factor', 'day', 'hour_cluster_min', 'hour_cluster_mean', 'month_cluster_mean',
                                  'month_cluster_min', 'stories_cluster_mean', 'stories_cluster_min', 'tas', 'tdew', 'rh', 'pressure', 'Horizontal Infrared Radiation Intensity',
                                  'Global Horizontal Radiation', 'Direct Normal Radiation', 'Diffuse Horizontal Radiation', 'wind', 'wind speed', 'Total Sky Cover', 'Opaque Sky Cover',
                                  'lon', 'lat', 'high', 'location_cluster_coldB', 'location_cluster_severe coldA', 'location_cluster_severe coldB',
                                  'location_cluster_severe coldC', 'location_cluster_warm', 'location_cluster_winter cold', 'location_cluster_winter warm']

# List of variables used for cooling model training
Model_training_variable_cooling = ['roof_u-factor', 'wall_u-factor', 'window_u-factor', 'day', 'hour_cluster_min', 'hour_cluster_mean',
                                   'month_cluster_mean', 'month_cluster_min', 'stories_cluster_min', 'tas', 'tdew', 'rh', 'pressure', 'Horizontal Infrared Radiation Intensity',
                                   'Global Horizontal Radiation', 'Direct Normal Radiation', 'Diffuse Horizontal Radiation', 'wind',
                                   'wind speed', 'Total Sky Cover', 'Opaque Sky Cover', 'lon', 'lat', 'high', 'location_cluster_coldB', 'location_cluster_severe coldA', 'location_cluster_severe coldB',
                                   'location_cluster_severe coldC', 'location_cluster_warm', 'location_cluster_winter cold', 'location_cluster_winter warm']

# List of all variables used in the models
variables_name = ['roof_u-factor', 'wall_u-factor', 'window_u-factor', 'stories', 'month', 'day', 'hour', 'tas', 'tdew', 'rh', 'pressure', 'Horizontal Infrared Radiation Intensity',
                  'Global Horizontal Radiation', 'Direct Normal Radiation', 'Diffuse Horizontal Radiation', 'wind', 'wind speed', 'Total Sky Cover', 'Opaque Sky Cover',
                  'location', 'lon', 'lat', 'high', 'heatingenergy', 'coolingenergy']


# Read in the CSV file containing city information into a pandas DataFrame
city_information_table = pd.read_csv('Information of 1667 cities.csv')


# Define a function to read an Excel file given its path and sheet index
def read_excel(path, index):
    # Open the Excel file
    book = xlrd.open_workbook(path)
    # Select the sheet by index
    sheet = book.sheet_by_index(index)
    # Get the number of rows in the sheet
    nrows = sheet.nrows
    # Get the number of columns in the sheet
    ncols = sheet.ncols
    # Return the sheet object, number of rows, and number of columns
    return sheet, nrows, ncols

# Define a function to copy a file from one path to another
def File_SaveAs(f1, f2):
    # Open the source file in read-binary mode
    with open(f1, 'rb') as fp1:
        # Read the contents of the file into a byte string
        b1 = fp1.read()
    # Open the destination file in write-binary mode
    with open(f2, 'wb') as fp2:
        # Write the contents of the source file to the destination file
        fp2.write(b1)


# Process the upper and lower limits of sampling variables to generate a specified number of random samples
def Sample_input_data_process (sample_number, sample_limit):
    # Obtaining sample range
    coefficient_lower = np.zeros((sample_number, 2))  # initialize coefficient lower array
    coefficient_upper = np.zeros((sample_number, 2))  # initialize coefficient upper array
    for i in range(sample_number):
        coefficient_lower[i, 0] = 1 - i / sample_number  # calculate lower coefficient values
        coefficient_lower[i, 1] = i / sample_number  # calculate lower coefficient values
    for i in range(sample_number):
        coefficient_upper[i, 0] = 1-(i+1) / sample_number  # calculate upper coefficient values
        coefficient_upper[i, 1] = (i+1) / sample_number  # calculate upper coefficient values
    sample_limit_lower = coefficient_lower @ sample_limit.T  # calculate lower limit for sample range
    sample_limit_upper = coefficient_upper @ sample_limit.T  # calculate upper limit for sample range
    sample_range = np.dstack((sample_limit_lower.T, sample_limit_upper.T))  # stack lower and upper limits

    # Random sampling in the sample range
    sample_variable_number = sample_range.shape[0]  # number of variables to sample
    sample_layer_number  = sample_range.shape[1]  # number of layers to sample
    coefficient_random = np.zeros((sample_variable_number,sample_layer_number, 2))  # initialize coefficient random array
    sample_matrix = np.zeros((sample_layer_number, sample_variable_number))  # initialize sample matrix

    for m in range(sample_variable_number):
        for i in range(sample_layer_number):
            y = random.random()  # generate a random number
            coefficient_random[m, i, 0] = 1 - y  # calculate lower coefficient values
            coefficient_random[m, i, 1] = y  # calculate upper coefficient values
    temp_arr = sample_range * coefficient_random  # generate temporary array

    for j in range(sample_variable_number):
        temp_random = temp_arr[j, :, 0] + temp_arr[j, :, 1]  # calculate random values within the range
        sample_matrix[:,j] = temp_random  # add random values to the sample matrix

    for n in range(sample_matrix.shape[1]):
        np.random.shuffle(sample_matrix[:, n])  # disrupt the order of the values in each column
    return sample_matrix


# For all the processes related to the sample, only one parameter is left, that is, sample_number
def Sample(sample_number):
    # Given sampled variables and variable limits
    sample_limit = np.array([[0.08, 0.08, 0.25, 0.5, 0.5, 0.5, 0.5],
                          [1.98, 1.5, 3.75, 10.5, 270.5, 8760.5, 1752.5]]).T
    sample_name = ["roof_u-factor", "wall_u-factor", "window_u-factor", "stories", 'location', 'extract_start', 'extract_interval']
    sample_matrix = Sample_input_data_process(sample_number, sample_limit)
    sample_output_data = pd.DataFrame(sample_matrix, columns = sample_name)

    # Keep the decimals as desired for each variable
    sample_output_data['roof_u-factor'] = sample_output_data['roof_u-factor'].round(3)
    sample_output_data['wall_u-factor'] = sample_output_data['wall_u-factor'].round(3)
    sample_output_data['window_u-factor'] = sample_output_data['window_u-factor'].round(3)
    sample_output_data['stories'] = sample_output_data['stories'].round(0)
    sample_output_data['location'] = sample_output_data['location'].round(0)
    sample_output_data['extract_start'] = sample_output_data['extract_start'].round(0)
    sample_output_data['extract_interval'] = sample_output_data['extract_interval'].round(0)
    #sample_output_data.to_excel('./sample_output.xls')


# Process the files used for simulation according to the parameters specified in the sample
def Simulate_input_data_process(sample_order):
    # Read the sample set generated by sampling and process it sample by sample
    path_sample_output = './sample_output.xls'
    sheet_sample_output, nrows_sample_output, ncols_sample_output = read_excel(path_sample_output, 0)

    # Select the weather file specified by the sample, and change it to a unified simulation file
    simulate_input_weather_station = int(sheet_sample_output.cell_value(sample_order, 5))
    path_epwfile = './weather_energyplus/{simulate_input_weather_station}_weather_site.epw'.format(simulate_input_weather_station = simulate_input_weather_station)
    path_simulate_input_weather = './overall/new.epw'
    File_SaveAs(path_epwfile, path_simulate_input_weather)

    # Select the building file specified by the sample, and change it to a unified simulation file
    path_roof_ufactor = 'roof_u-factor.xls'
    path_wall_ufactor = 'wall_u-factor.xls'
    path_window_ufactor = 'window_u-factor.xls'
    sheet_roof_ufactor, nrows_roof_ufactor, ncols_roof_ufactor = read_excel(path_roof_ufactor, 0)
    sheet_wall_ufactor, nrows_wall_ufactor, ncols_wall_ufactor = read_excel(path_wall_ufactor, 0)
    sheet_window_ufactor, nrows_window_ufactor, ncols_window_ufactor = read_excel(path_window_ufactor, 0)

    sample_envelopes = []
    for col in range(1, 4):
        sample_envelopes.append(sheet_sample_output.cell_value(sample_order, col))
    simulate_input_building_story = ('%.0f' % float(sheet_sample_output.cell_value(sample_order, 4)))
    IDF.setiddname(iddfile)
    simulate_input_building = IDF('./prototype/simpletrain{simulate_input_building_story}.idf'.format(simulate_input_building_story = simulate_input_building_story))

    # Modify the heat transfer coefficients of the three main constructions of the building envelope
    material_roof = simulate_input_building.idfobjects['MATERIAL'][6]
    for row_roof_ufactor in range(1, nrows_roof_ufactor):
        if ('%.3f' %float(sheet_roof_ufactor.cell_value(row_roof_ufactor, 1))) == ('%.3f' %float(sample_envelopes[0])):
            ufactor_roof = sheet_roof_ufactor.cell_value(row_roof_ufactor, 0)
    material_roof.Conductivity = ufactor_roof

    material_wall = simulate_input_building.idfobjects['MATERIAL'][3]
    for row_wall_ufactor in range(1, nrows_wall_ufactor):
        if ('%.3f' %float(sheet_wall_ufactor.cell_value(row_wall_ufactor, 1))) == ('%.3f' %float(sample_envelopes[1])):
            ufactor_wall = sheet_wall_ufactor.cell_value(row_wall_ufactor, 0)
    material_wall.Conductivity = ufactor_wall

    material_window = simulate_input_building.idfobjects['WindowMATERIAL:Glazing'][0]
    for row_window_ufactor in range(1, nrows_window_ufactor):
        if ('%.3f' %float(sheet_window_ufactor.cell_value(row_window_ufactor, 1))) == ('%.3f' %float(sample_envelopes[2])):
            ufactor_window = sheet_window_ufactor.cell_value(row_window_ufactor, 0)
    material_window.Conductivity = ufactor_window
    material_window.Solar_Transmittance_at_Normal_Incidence = 0.5

    # Add the ddy file for each locale to the building file
    ddyidf = IDF('./prototype/weather_energyplus/{simulate_input_weather_station}_weather_site.ddy'.format(simulate_input_weather_station=simulate_input_weather_station))
    designdays = ddyidf.idfobjects["SizingPeriod:DesignDay".upper()]
    designday_cooling = designdays[6]
    designday_heating = designdays[0]
    # copy a design day into your idf file
    simulate_input_building.copyidfobject(designday_cooling)
    simulate_input_building.copyidfobject(designday_heating)
    simulate_input_building.saveas('prototype/new/new.idf')

    return simulate_input_building, simulate_input_building_story


# Run the simulation software and get the energy consumption intensity file of the simulation output
def Simulate(sample_order):
    # Process the input data for simulation
    simulate_input_building, simulate_input_building_story = Simulate_input_data_process(sample_order)

    # Find and click the "Simulate" button, then sleep for a certain amount of time
    simulate_location_begin = pyautogui.locateOnScreen(image = 'prototype/new/simulate.png', confidence = 0.56)
    simulate_location_begin_x, simulate_location_begin_y = pyautogui.center(simulate_location_begin)
    pyautogui.moveTo(simulate_location_begin_x, simulate_location_begin_y, duration = 1)
    pyautogui.click()
    time.sleep(Simulate_sleep_times[int(simulate_input_building_story)])

    # Find and click the "OK" button to end the simulation
    simulate_location_end = pyautogui.locateOnScreen(image = 'prototype/new/ok.png', confidence = 0.7)
    while simulate_location_end is None:
        simulate_location_end = pyautogui.locateOnScreen(image = 'prototype/new/ok.png', confidence = 0.7)
    simulate_location_end_x, simulate_location_end_y = pyautogui.center(simulate_location_end)
    pyautogui.moveTo(simulate_location_end_x, simulate_location_end_y, duration = 1)
    pyautogui.click()

    # Read the simulation output data and save it to a csv file
    dtf_simulate_output = pd.read_csv('prototype/new/new.csv')
    dtf_simulate_output.to_csv('./simulate_output_intensity/coldA_{sample_order}.csv'.format(sample_order = sample_order))


# The collation of data into a standardized data format that can be used for training models
def Train_input_data_produce(sample_order):
    # Read the sampling results of the Latin hypercube
    path_sample_output = './sample_output.xls'
    sheet_sample_output, nrows_sample_output, ncols_sample_output = read_excel(path_sample_output, 0)

    # Read all seven variables sampled line by line
    train_samples = []
    for row in range(1, sample_order):
        print(row)
        sample_envelopes = []  # Create an empty list to store the envelope information of the current sample
        sample_story = int(sheet_sample_output.cell_value(row,
                                                          4))  # Get the number of stories for the current sample from the Excel sheet
        sample_order = int(
            sheet_sample_output.cell_value(row, 0)) + 1  # Get the order of the current sample from the Excel sheet
        sample_station = int(sheet_sample_output.cell_value(row,
                                                            5))  # Get the weather station number for the current sample from the Excel sheet
        sample_simulate_output_start = int(sheet_sample_output.cell_value(row,
                                                                          6))  # Get the starting time point for the simulation output for the current sample from the Excel sheet
        sample_simulate_output_interval = int(sheet_sample_output.cell_value(row,
                                                                             7))  # Get the time interval for the simulation output for the current sample from the Excel sheet

        # Store the envelope information of the current sample in the sample_envelopes list
        sample_envelopes.append(sheet_sample_output.cell_value(row, 1))
        sample_envelopes.append(sheet_sample_output.cell_value(row, 2))
        sample_envelopes.append(sheet_sample_output.cell_value(row, 3))
        sample_envelopes.append(sample_story)

        path_train_input_weather = './train_input_weather/{sample_station}_weather_site_hours.csv'.format(
            sample_station=int(sample_station))
        path_simulate_output_intensity = './simulate_output_intensity/coldA_{sample_order}.csv'.format(
            sample_order=sample_order)
        dtf_train_input_weather = pd.read_csv(path_train_input_weather)
        dtf_train_input_intensity = pd.read_csv(path_simulate_output_intensity)
        dtf_train_input_weather = dtf_train_input_weather.drop('Unnamed: 0', axis=1)
        dtf_train_input_intensity = dtf_train_input_intensity.drop('Unnamed: 0', axis=1)
        dtf_train_input_intensity = dtf_train_input_intensity.drop('Date/Time', axis=1)
        dtf_train_input_intensity = pd.DataFrame(dtf_train_input_intensity.values.T,
                                                 index=dtf_train_input_intensity.columns,
                                                 columns=dtf_train_input_intensity.index)

        # Calculate the heating and cooling loads for each hour of the current sample and store them in two separate DataFrames
        dtf_train_input_heating = dtf_train_input_intensity[:sample_story * 4].apply(lambda x: x.sum(),
                                                                                     axis=0) / sample_story / 783.65
        dtf_train_input_cooling = dtf_train_input_intensity[sample_story * 4:].apply(lambda x: x.sum(),
                                                                                     axis=0) / sample_story / 783.65

        for j in range(5):  # Iterate through the five output time points for each sample
            sample_simulate_output_point = sample_simulate_output_start + j * sample_simulate_output_interval  # Calculate the current output time point
            if sample_simulate_output_point > 8760:  # If the current output time point is beyond one year, subtract 8760 to keep it within the range of one year
                sample_simulate_output_point = sample_simulate_output_point - 8760
            train_sample = pd.concat(
                [dtf_train_input_weather[sample_simulate_output_point - 1: sample_simulate_output_point],
                 dtf_train_input_heating[sample_simulate_output_point - 1: sample_simulate_output_point],
                 dtf_train_input_cooling[sample_simulate_output_point - 1: sample_simulate_output_point]], axis=1,
                join='inner')
            train_sample = train_sample.reset_index()
            train_sample = train_sample.drop('index', axis=1)
            train_sample = pd.concat([pd.DataFrame(pd.DataFrame(sample_envelopes).values.T), train_sample], axis=1,
                                     join='inner')
            train_samples.extend(np.array(train_sample).tolist())

    # Combine all the training data samples into a pandas dataframe and save it to a csv file
    train_samples = pd.DataFrame(train_samples)
    train_samples.columns = variables_name
    train_samples.to_csv('./train_input.csv')


# Analysis of the relationship between independent variables and target variables in training samples
def Train_input_data_sanalyze(dtf):
    # Univariate analysis
    freqdist_plot(dtf, "Y", box_logscale=True, figsize=(10,5)) # Plot the frequency distribution of the target variable 'Y' on a logarithmic scale
    freqdist_plot(dtf, "stories", figsize=(5,3)) # Plot the frequency distribution of the feature 'stories'

    # Binary distribution analysis
    bivariate_plot(dtf, x="roof_u-factor", y="Y", figsize=(15,5)) # Plot the bivariate distribution of 'roof_u-factor' and 'Y'
    bivariate_plot(dtf, x="wall_u-factor", y="Y", figsize=(15,5)) # Plot the bivariate distribution of 'wall_u-factor' and 'Y'
    bivariate_plot(dtf, x="window_u-factor", y="Y", figsize=(15,5)) # Plot the bivariate distribution of 'window_u-factor' and 'Y'
    bivariate_plot(dtf, x="stories", y="Y", figsize=(15,5)) # Plot the bivariate distribution of 'stories' and 'Y'
    bivariate_plot(dtf, x="tas", y="Y", figsize=(15,5)) # Plot the bivariate distribution of 'tas' and 'Y'
    bivariate_plot(dtf, x="tdew", y="Y", figsize=(15,5)) # Plot the bivariate distribution of 'tdew' and 'Y'
    bivariate_plot(dtf, x="rh", y="Y", figsize=(15,5)) # Plot the bivariate distribution of 'rh' and 'Y'
    bivariate_plot(dtf, x="pressure", y="Y", figsize=(15,5)) # Plot the bivariate distribution of 'pressure' and 'Y'
    bivariate_plot(dtf, x='Horizontal Infrared Radiation Intensity', y="Y", figsize=(15, 5)) # Plot the bivariate distribution of 'Horizontal Infrared Radiation Intensity' and 'Y'
    bivariate_plot(dtf, x='Global Horizontal Radiation', y="Y", figsize=(15, 5)) # Plot the bivariate distribution of 'Global Horizontal Radiation' and 'Y'
    bivariate_plot(dtf, x='Direct Normal Radiation', y="Y", figsize=(15, 5)) # Plot the bivariate distribution of 'Direct Normal Radiation' and 'Y'
    bivariate_plot(dtf, x='Diffuse Horizontal Radiation', y="Y", figsize=(15, 5)) # Plot the bivariate distribution of 'Diffuse Horizontal Radiation' and 'Y'
    bivariate_plot(dtf, x='wind', y="Y", figsize=(15, 5)) # Plot the bivariate distribution of 'wind' and 'Y'
    bivariate_plot(dtf, x='wind speed', y="Y", figsize=(15, 5)) # Plot the bivariate distribution of 'wind speed' and 'Y'
    bivariate_plot(dtf, x='Total Sky Cover', y="Y", figsize=(15, 5)) # Plot the bivariate distribution of 'Total Sky Cover' and 'Y'
    bivariate_plot(dtf, x='Opaque Sky Cover', y="Y", figsize=(15, 5)) # Plot the bivariate distribution of 'Opaque Sky Cover' and 'Y'
    bivariate_plot(dtf, x="month", y="Y", figsize=(15,5)) # Plot the bivariate distribution of 'month' and 'Y'
    bivariate_plot(dtf, x="hour", y="Y", figsize=(15, 5))  # plot Y against hour
    bivariate_plot(dtf, x="day", y="Y", figsize=(15, 5))  # plot Y against day
    bivariate_plot(dtf, x="location", y="Y", figsize=(15, 5))  # plot Y against location
    bivariate_plot(dtf, x="lon", y="Y", figsize=(15, 5))  # plot Y against longitude
    bivariate_plot(dtf, x="lat", y="Y", figsize=(15, 5))  # plot Y against latitude
    bivariate_plot(dtf, x="high", y="Y", figsize=(15, 5))  # plot Y against high

    # Relevance and significance analysis
    corr = corr_matrix(dtf, method="pearson", negative=False, annotation=True,
                       figsize=(15, 7))  # compute correlation matrix
    dic_feat_sel = features_selection(dtf, y="Y", task="regression", top=10,
                                      figsize=(10, 5))  # select top 10 features using regression task
    model = ensemble.RandomForestRegressor(n_estimators=100, criterion="mse",
                                           random_state=0)  # create a random forest regressor model
    feat_imp = features_importance(X=dtf.drop("Y", axis=1).values, y=dtf["Y"].values,
                                   X_names=dtf.drop("Y", axis=1).columns.tolist(), model=model, task="regression",
                                   figsize=(15, 5))  # compute feature importance using the random forest model


# The preprocessing of the training input data returns the processed data and scaling parameters
def Train_input_data_process():
    # Reading training input data
    dtf_train_input = pd.read_csv('./train.csv')
    for index,row in dtf_train_input.iterrows():
        if row['heatingenergy']!=0 and row['coolingenergy']!=0:
            dtf_train_input = dtf_train_input.drop(index)

    # Different ways to merge data for different energy consumption conditions
    # hours
    hour_clusters_heating = {"max": [], "mean": [20, 21, 22, 23, 24, 11, 12], "min": [13, 14, 15, 16, 17, 18, 19]}
    hour_clusters_cooling = {"max": [14, 15, 16, 17, 18, 19, 20, 21], "mean": [11, 12, 13, 22, 23, 24], "min": []}
    # months
    month_clusters_heating = {"max": [1, 12], "mean": [2, 11], "min": []}
    month_clusters_cooling = {"max": [6, 7, 8], "mean": [5, 9], "min": []}
    # city_building_stories
    city_building_stories_clusters_heating = {"max": [1], "mean": [2, 3], "min": [4, 5, 6, 7, 8, 9, 10]}
    city_building_stories_clusters_cooling = {"max": [], "min": [1]}
    # location
    location_clusters = {
        "coldA": [29, 32, 33, 35, 36, 39, 41, 73, 75, 76, 135, 136, 137, 138, 141, 143, 162, 174, 175,
                  176, 184, 191, 196, 207, 210, 211, 212, 216, 218, 219, 221, 222, 225, 226, 227, 229,
                  231, 233, 235, 237, 240, 241, 242, 244, 250, 251, 252, 253, 254],
        "coldB": [3, 9, 10, 59, 60, 62, 64, 77, 78, 79, 121, 122, 125, 206, 208, 209, 213, 220, 224,
                  228, 230, 243, 246, 247],
        "severe coldA": [38, 89, 93, 95, 99, 146, 150, 157, 167, 171, 178, 180, 183, 186, 187],
        "severe coldB": [30, 80, 84, 85, 86, 87, 88, 90, 91, 92, 94, 96, 97, 98, 100, 115, 145, 152, 154,
                         156, 161, 169, 172, 173, 177, 179, 182, 185, 188, 190, 194, 198, 232, 234, 236],
        "severe coldC": [31, 34, 40, 74, 101, 112, 113, 114, 116, 117, 118, 119, 134, 139, 140, 142, 144,
                         147, 148, 149, 151, 153, 155, 158, 159, 160, 163, 164, 165, 166, 168, 170, 181,
                         189, 193, 202, 223, 238, 239, 245, 248, 249, 256],
        "winter cold": [],
        "winter warm": [13, 14, 18, 20, 21, 22, 24, 25, 27, 28, 42, 43, 45, 46, 47, 48, 49, 50, 51, 81,
                        82, 83, 261, 265],
        "warm": [52, 53, 56, 57, 195, 204, 255, 257, 258, 259, 260, 262, 263, 264]}

    # Determine the scaling rule
    # RobustScaler transforms features by subtracting the median and then dividing by the interquartile range(75 % to 25 % value).
    RobustScalerX_heating = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
    RobustScalerY_heating = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
    RobustScalerX_cooling = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
    RobustScalerY_cooling = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))

    # disposal data for heating
    dtf_train_input_heating = dtf_train_input.rename(columns={"heatingenergy": "Y"})
    dtf_train_input_heating = dtf_train_input_heating.drop('coolingenergy', axis=1)

    # data clusters of heating
    dtf_train_input_heating = add_feature_clusters(dtf_train_input_heating, "hour",
                                                   dic_clusters_mapping=hour_clusters_heating, dropx=False)
    dtf_train_input_heating = add_dummies(dtf_train_input_heating, x="hour_cluster", dropx=True)
    dtf_train_input_heating = add_feature_clusters(dtf_train_input_heating, "month",
                                                   dic_clusters_mapping=month_clusters_heating, dropx=False)
    dtf_train_input_heating = add_dummies(dtf_train_input_heating, x="month_cluster", dropx=True)
    dtf_train_input_heating = add_feature_clusters(dtf_train_input_heating, "stories",
                                                   dic_clusters_mapping=city_building_stories_clusters_heating,
                                                   dropx=False)
    dtf_train_input_heating = add_dummies(dtf_train_input_heating, x="stories_cluster", dropx=True)
    dtf_train_input_heating = add_feature_clusters(dtf_train_input_heating, "location",
                                                   dic_clusters_mapping=location_clusters, dropx=False)
    dtf_train_input_heating = add_dummies(dtf_train_input_heating, x="location_cluster", dropx=True)

    # drop data for heating
    dtf_train_input_heating = dtf_train_input_heating.drop('hour', axis=1)
    dtf_train_input_heating = dtf_train_input_heating.drop('month', axis=1)
    dtf_train_input_heating = dtf_train_input_heating.drop('stories', axis=1)
    dtf_train_input_heating = dtf_train_input_heating.drop('location', axis=1)

    # Calculate the scaling parameters of heating
    dtf_train_input_heating, scalerX_heating, scalerY_heating = scaling(dtf_train_input_heating, y="Y",
                                                                        scalerX=RobustScalerX_heating,
                                                                        scalerY=RobustScalerY_heating,
                                                                        task="regression")

    # disposal data for cooling
    dtf_train_input_cooling = dtf_train_input.rename(columns={"coolingenergy": "Y"})
    dtf_train_input_cooling = dtf_train_input_cooling.drop('heatingenergy', axis=1)

    # data clusters of cooling
    dtf_train_input_cooling = add_feature_clusters(dtf_train_input_cooling, "hour",
                                                   dic_clusters_mapping=hour_clusters_cooling, dropx=False)
    dtf_train_input_cooling = add_dummies(dtf_train_input_cooling, x="hour_cluster", dropx=True)
    dtf_train_input_cooling = add_feature_clusters(dtf_train_input_cooling, "month",
                                                   dic_clusters_mapping=month_clusters_cooling, dropx=False)
    dtf_train_input_cooling = add_dummies(dtf_train_input_cooling, x="month_cluster", dropx=True)
    dtf_train_input_cooling = add_feature_clusters(dtf_train_input_cooling, "stories",
                                                   dic_clusters_mapping=city_building_stories_clusters_cooling,
                                                   dropx=False)
    dtf_train_input_cooling = add_dummies(dtf_train_input_cooling, x="stories_cluster", dropx=True)
    dtf_train_input_cooling = add_feature_clusters(dtf_train_input_cooling, "location",
                                                   dic_clusters_mapping=location_clusters, dropx=False)
    dtf_train_input_cooling = add_dummies(dtf_train_input_cooling, x="location_cluster", dropx=True)

    # drop data for cooling
    dtf_train_input_cooling = dtf_train_input_cooling.drop('hour', axis=1)
    dtf_train_input_cooling = dtf_train_input_cooling.drop('month', axis=1)
    dtf_train_input_cooling = dtf_train_input_cooling.drop('stories', axis=1)
    dtf_train_input_cooling = dtf_train_input_cooling.drop('location', axis=1)

    # Calculate the scaling parameters of cooling
    dtf_train_input_cooling, scalerX_cooling, scalerY_cooling = scaling(dtf_train_input_cooling, y="Y",
                                                                        scalerX=RobustScalerX_cooling,
                                                                        scalerY=RobustScalerY_cooling,
                                                                        task="regression")

    return dtf_train_input_heating, dtf_train_input_cooling, scalerX_heating, scalerY_heating, scalerX_cooling, scalerY_cooling


# Use the GBR training model
def Train_surrogate_model(dtf_train_split, Model_training_variable):

    dtf_train_x = dtf_train_split[Model_training_variable].values
    dtf_train_y = dtf_train_split["Y"].values

    my_model = ensemble.GradientBoostingRegressor()

    param_dic = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001],      #weighting factor for the corrections by new trees when added to the model
                     'n_estimators':[100,250,500,750,1000,1250,1500,1750],  #number of trees added to the model
                     'max_depth':[2,3,4,5,6,7],                             #maximum depth of the tree
                     'min_samples_split':[2,4,6,8,10,20,40,60,100],         #sets the minimum number of samples to split
                     'min_samples_leaf':[1,3,5,7,9],                        #the minimum number of samples to form a leaf
                     'max_features':[2,3,4,5,6,7],                          #square root of features is usually a good starting point
                     'subsample':[0.7,0.75,0.8,0.85,0.9,0.95,1]}            #the fraction of samples to be used for fitting the individual base learners. Values lower than 1 gen

    my_model = tune_regr_model(dtf_train_x, dtf_train_y, my_model, param_dic, scoring = "r2", searchtype = "RandomSearch", n_iter = 100, cv = 5, figsize = (10,5))#应该是模型优选的一个过程，

    return my_model


# The trained model is tested using a segmented test set
def Train_surrogate_model_test(my_model, dtf_train_split, dtf_test_split, scalerY, Model_training_variable, HVAC):
    # Extract the input and output variables from the training and test data
    dtf_train_x = dtf_train_split[Model_training_variable].values
    dtf_train_y = dtf_train_split["Y"].values
    dtf_test_x = dtf_test_split[Model_training_variable].values
    dtf_test_y = dtf_test_split["Y"].values

    # Use the surrogate model to make predictions on the test data
    prediction = my_model.predict(dtf_test_x)
    # Transform the predicted and true output values to their original scale
    prediction = scalerY.inverse_transform(prediction.reshape(-1, 1)).reshape(-1)/1000
    true = dtf_test_y
    true = scalerY.inverse_transform(true.reshape(-1, 1)).reshape(-1)/1000
    # Calculate the R-squared score of the model's predictions
    score = metrics.r2_score(true, prediction)
    # Create a scatter plot to visualize the model's accuracy on the test data
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(prediction, true, lw=2, alpha=0.5, label='R2 = %0.2f' % (score))
    ax.plot([min(true), max(true)], [min(true), max(true)], linestyle='--', lw=2, color='black')
    ax.set_xlabel('Predicted (Unit : KJ/m\u00b2*h)')
    ax.set_ylabel('True (Unit : KJ/m\u00b2*h)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title('Accuracy detection on the test set')
    ax.legend(frameon=False)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.93, wspace=0, hspace=0)
    # Save the scatter plot as a PNG file
    plt.savefig('picture/new/{hvac}_testset.png'.format(hvac = HVAC))
    #plt.show()



# For all the processes related to the training samples, only one parameter is left, that is, HVAC
def Train(HVAC):
    # Produce training input data
    # Train_input_data_produce(number)

    # Read and preprocess data for training
    dtf_train_input_heating, dtf_train_input_cooling, scalerX_heating, scalerY_heating, scalerX_cooling, scalerY_cooling = Train_input_data_process()
    dtf_train_split_heating, dtf_test_split_heating = dtf_partitioning(dtf_train_input_heating, y="Y", test_size=0.3, shuffle=False)
    dtf_train_split_cooling, dtf_test_split_cooling = dtf_partitioning(dtf_train_input_cooling, y="Y", test_size=0.3, shuffle=False)

    if HVAC == 'heating':
        Model_training_variable = Model_training_variable_heating
        dtf_train_input, dtf_train_split, dtf_test_split, scalerY = dtf_train_input_heating, dtf_train_split_heating, dtf_test_split_heating, scalerY_heating
    elif HVAC == 'cooling':
        Model_training_variable = Model_training_variable_cooling
        dtf_train_input, dtf_train_split, dtf_test_split, scalerY = dtf_train_input_cooling, dtf_train_split_cooling, dtf_test_split_cooling, scalerY_cooling

    # Analyze the training input data
    # Train_input_data_sanalyze(dtf_train_input)

    # Train, test, and save surrogate models
    path_surrogate_model = './model_{hvac}.pkl'.format(hvac=HVAC)
    my_model = Train_surrogate_model(dtf_train_split, Model_training_variable)
    # my_model = joblib.load('./model_{hvac}_10000-51.pkl'.format(hvac=HVAC))
    Train_surrogate_model_test(my_model, dtf_train_split, dtf_test_split, scalerY, Model_training_variable, HVAC)
    joblib.dump(my_model, path_surrogate_model)


# Read and collate the predicted input data
def Predict_input_data_produce(w_scenario, e_path, year_observe, vintage, city_climate_station, city_climate_zone, city_building_stories):

    # Read the weather part and envelope part of the forecast input data
    path_weather_predict = './weather_scenario/{city_climate_station}_weather_site/{city_climate_station}-{w_scenario}-{year}.csv'.format(
        w_scenario = w_scenario, year = year_observe, city_climate_station = city_climate_station)
    path_envelope_predict = './envelope_path/{city_climate_zone}/{city_climate_zone}-{city_building_stories}-{e_path}-{year}-{age}-envelope.csv'.format(
        e_path = e_path, year = year_observe, age = int(vintage), city_climate_zone = city_climate_zone, city_building_stories = city_building_stories)
    dtf_predict_input_envelope = pd.read_csv(path_envelope_predict)
    dtf_predict_input_envelope = dtf_predict_input_envelope.drop('Unnamed: 0', axis=1)
    dtf_predict_input_weather = pd.read_csv(path_weather_predict)

    # The merging of two parts of data
    dtf_predict_input = pd.concat([dtf_predict_input_envelope, dtf_predict_input_weather], axis=1, join='inner')
    dtf_predict_input = dtf_predict_input.drop('Unnamed: 0', axis=1)
    # Returns the processed merged data
    return dtf_predict_input


# Standardization processes predictive input data
def Predict_input_data_process(city_building_hvac, scalerX, dtf_predict_input):
    # Due to the singularity of climate stations(has been pre-written) and stories, the standardized processing of prediction input data is different from that of training input data
    # hours
    hour_clusters_heating = {"max": [], "mean": [20, 21, 22, 23, 24, 11, 12], "min": [13, 14, 15, 16, 17, 18, 19]}
    hour_clusters_cooling = {"max": [14, 15, 16, 17, 18, 19, 20, 21], "mean": [11, 12, 13, 22, 23, 24], "min": []}
    # months
    month_clusters_heating = {"max": [1, 12], "mean": [2, 11], "min": []}
    month_clusters_cooling = {"max": [6, 7, 8], "mean": [5, 9], "min": []}

    # Check if HVAC system is heating or cooling
    if city_building_hvac == 'heating':
        # If heating, use heating-specific model training variables
        Model_training_variable = Model_training_variable_heating

        # Cluster hours of the day and add dummies for each cluster
        dtf_predict_input_heating = add_feature_clusters(dtf_predict_input, "hour",
                                                         dic_clusters_mapping=hour_clusters_heating, dropx=False)
        dtf_predict_input_heating = add_dummies(dtf_predict_input_heating, x="hour_cluster", dropx=True)

        # Cluster months of the year and add dummies for each cluster
        dtf_predict_input_heating = add_feature_clusters(dtf_predict_input_heating, "month",
                                                         dic_clusters_mapping=month_clusters_heating, dropx=False)
        dtf_predict_input = add_dummies(dtf_predict_input_heating, x="month_cluster", dropx=True)

        # Create a new DataFrame for the building stories and fill with appropriate cluster values based on number of stories
        if dtf_predict_input['stories'][0] == 1:
            data_city_building_stories = pd.concat(
                [pd.DataFrame(np.repeat(0, 8760, axis=0)), pd.DataFrame(np.repeat(0, 8760, axis=0))], axis=1)
            data_city_building_stories.columns = ['stories_cluster_mean', 'stories_cluster_min']
        elif dtf_predict_input['stories'][0] == 2 or dtf_predict_input['stories'][0] == 3:
            data_city_building_stories = pd.concat(
                [pd.DataFrame(np.repeat(1, 8760, axis=0)), pd.DataFrame(np.repeat(0, 8760, axis=0))], axis=1)
            data_city_building_stories.columns = ['stories_cluster_mean', 'stories_cluster_min']
        else:
            data_city_building_stories = pd.concat(
                [pd.DataFrame(np.repeat(0, 8760, axis=0)), pd.DataFrame(np.repeat(1, 8760, axis=0))], axis=1)
            data_city_building_stories.columns = ['stories_cluster_mean', 'stories_cluster_min']

    elif city_building_hvac == 'cooling':
        # If cooling, use cooling-specific model training variables
        Model_training_variable = Model_training_variable_cooling

        # Cluster hours of the day and add dummies for each cluster
        dtf_predict_input_cooling = add_feature_clusters(dtf_predict_input, "hour",
                                                         dic_clusters_mapping=hour_clusters_cooling, dropx=False)
        dtf_predict_input_cooling = add_dummies(dtf_predict_input_cooling, x="hour_cluster", dropx=True)

        # Cluster months of the year and add dummies for each cluster
        dtf_predict_input_cooling = add_feature_clusters(dtf_predict_input_cooling, "month",
                                                         dic_clusters_mapping=month_clusters_cooling, dropx=False)
        dtf_predict_input = add_dummies(dtf_predict_input_cooling, x="month_cluster", dropx=True)

        # Create a new DataFrame for the building stories and fill with appropriate cluster values based on number of stories
        if dtf_predict_input['stories'][0] == 1:
            data_city_building_stories = pd.DataFrame(np.repeat(1, 8760, axis=0))
            data_city_building_stories.columns = ['stories_cluster_min']
        else:
            data_city_building_stories = pd.DataFrame(np.repeat(0, 8760, axis = 0))
            data_city_building_stories.columns = ['stories_cluster_min']

    dtf_predict_input = pd.concat([dtf_predict_input, data_city_building_stories], axis = 1)
    dtf_predict_input = dtf_predict_input.drop('hour', axis = 1)
    dtf_predict_input = dtf_predict_input.drop('month', axis = 1)
    dtf_predict_input = dtf_predict_input.drop('stories', axis = 1)
    dtf_predict_input = dtf_predict_input.drop('location', axis = 1)

    # Standardized treatment
    dtf_predict_input_standard = scalerX.transform(dtf_predict_input)
    dtf_predict_input_standard = pd.DataFrame(dtf_predict_input_standard, columns = dtf_predict_input.columns, index = dtf_predict_input.index)
    dtf_predict_input = dtf_predict_input_standard[Model_training_variable].values

    return dtf_predict_input


# Categorize into three kinds for prediction
class Predict_surrogate_model:
    # Core model
    surrogate_model_heating = joblib.load('./model_heating.pkl')
    surrogate_model_cooling = joblib.load('./model_cooling.pkl')

    # The first type of prediction has the same city results directly replicated
    def Copy_predicted(self, city_code_predicted, city_code, city_building_hvacs):
        for city_building_hvac in city_building_hvacs:
            for e_path in Envelope_Paths:
                for w_scenario in Weather_Scenarios:
                    path_city_predicted = './energy/{city_code}/{city_code}-{Envelope_Path}-{Weather_Scenario}-{city_building_hvac}-tranist.xls'.format(
                        city_code = city_code_predicted, Envelope_Path = e_path, Weather_Scenario = w_scenario, city_building_hvac = city_building_hvac)
                    path_city = './energy/{city_code}/{city_code}-{Envelope_Path}-{Weather_Scenario}-{city_building_hvac}-tranist.xls'.format(
                        city_code = city_code, Envelope_Path = e_path, Weather_Scenario = w_scenario, city_building_hvac = city_building_hvac)
                    File_SaveAs(path_city_predicted, path_city)

    # The second type of forecasting is city forecasting with only one way of using energy
    def Single_predict(self,city_code, e_path, w_scenario, city_climate_station, city_building_stories, city_climate_zone, city_building_ages, city_building_hvac, city_building_hvac_time, scalerX_heating, scalerY_heating, scalerX_cooling, scalerY_cooling):
        dtf_predict_output_years_ages = []

        for year_observe in range(2020, 2101):
            print(e_path, w_scenario, year_observe)
            for vintage in city_building_ages:
                # Standardized processing of predictive input data
                dtf_predict_input = Predict_input_data_produce(w_scenario, e_path, year_observe, vintage, city_climate_station, city_climate_zone, city_building_stories)

                # The surrogate model is invoked energetically to predict
                if city_building_hvac == 'heating':
                    dtf_predict_input = Predict_input_data_process(city_building_hvac, scalerX_heating, dtf_predict_input)
                    dtf_predict_output = self.surrogate_model_heating.predict(dtf_predict_input)
                    dtf_predict_output = scalerY_heating.inverse_transform(dtf_predict_output.reshape(-1, 1)).reshape(-1)
                    dtf_predict_output_year_age_hours_heating = pd.DataFrame(dtf_predict_output)
                elif city_building_hvac == 'cooling':
                    dtf_predict_input = Predict_input_data_process(city_building_hvac, scalerX_cooling, dtf_predict_input)
                    dtf_predict_output = self.surrogate_model_cooling.predict(dtf_predict_input)
                    dtf_predict_output = scalerY_cooling.inverse_transform(dtf_predict_output.reshape(-1, 1)).reshape(-1)
                    dtf_predict_output_year_age_hours_cooling = pd.DataFrame(dtf_predict_output)

                # The annual energy intensity is calculated by way of using energy
                dtf_predict_output_year_ages = []
                if city_building_hvac == 'heating':
                    dtf_predict_output_year_age = pd.concat([dtf_predict_output_year_age_hours_heating[:int(city_building_hvac_time[0])+1], dtf_predict_output_year_age_hours_heating[int(city_building_hvac_time[1]):]], axis = 0, join='inner')
                elif city_building_hvac == 'cooling':
                    dtf_predict_output_year_age = dtf_predict_output_year_age_hours_cooling[int(city_building_hvac_time[0]): int(city_building_hvac_time[1])+1]
                dtf_predict_output_year_age[dtf_predict_output_year_age < 1000] = 0

                # Processing and storage of predictive output data
                dtf_predict_output_year_ages.append(dtf_predict_output_year_age.apply(lambda x: x.sum(), axis = 0).iloc[0])
                dtf_predict_output_year_ages.append(year_observe)
                dtf_predict_output_year_ages.append(vintage)
                dtf_predict_output_years_ages.append(dtf_predict_output_year_ages)
        dtf_predict_output_years_ages = pd.DataFrame(dtf_predict_output_years_ages)
        dtf_predict_output_years_ages.to_excel('./energy/{city_code}/{city_code}-{e_path}-{w_scenario}-{city_building_hvac}-tranist.xls'.format(city_code = city_code, e_path = e_path, w_scenario = w_scenario,city_building_hvac = city_building_hvac))

    # The third type of forecast is the city forecast which is used in both ways
    def Multiple_predict(self,city_code, e_path, w_scenario, city_climate_station, city_building_stories, city_climate_zone, city_building_ages, city_building_hvac_time, scalerX_heating, scalerY_heating, scalerX_cooling, scalerY_cooling):
        dtf_predict_output_years_ages_heating = []
        dtf_predict_output_years_ages_cooling = []
        
        for year_observe in range(2020, 2101):
            print(e_path, w_scenario, year_observe)
            for vintage in city_building_ages:
                # Standardized processing of predictive input data
                dtf_predict_input = Predict_input_data_produce(w_scenario, e_path, year_observe, vintage, city_climate_station, city_climate_zone, city_building_stories)
                dtf_predict_output_year_ages_heating = []
                dtf_predict_output_year_ages_cooling = []

                # The surrogate model is invoked energetically to predict
                dtf_predict_input_heating = Predict_input_data_process('heating', scalerX_heating, dtf_predict_input)
                dtf_predict_output_heating = self.surrogate_model_heating.predict(dtf_predict_input_heating)
                dtf_predict_output_heating = scalerY_heating.inverse_transform(dtf_predict_output_heating.reshape(-1, 1)).reshape(-1)
                dtf_predict_output_year_age_hours_heating = pd.DataFrame(dtf_predict_output_heating)
                dtf_predict_input_cooling = Predict_input_data_process('cooling', scalerX_cooling, dtf_predict_input)
                dtf_predict_output_cooling = self.surrogate_model_cooling.predict(dtf_predict_input_cooling)
                dtf_predict_output_cooling = scalerY_cooling.inverse_transform(dtf_predict_output_cooling.reshape(-1, 1)).reshape(-1)
                dtf_predict_output_year_age_hours_cooling = pd.DataFrame(dtf_predict_output_cooling)

                # The annual energy intensity is calculated by way of using energy
                dtf_predict_output_year_age_heating = pd.concat([dtf_predict_output_year_age_hours_heating[:int(city_building_hvac_time[0])+1], dtf_predict_output_year_age_hours_heating[int(city_building_hvac_time[1]):]], axis = 0, join = 'inner')
                dtf_predict_output_year_age_heating[dtf_predict_output_year_age_heating < 1000] = 0
                dtf_predict_output_year_age_cooling = dtf_predict_output_year_age_hours_cooling[int(city_building_hvac_time[2]): int(city_building_hvac_time[3]) + 1]
                dtf_predict_output_year_age_cooling[dtf_predict_output_year_age_cooling < 1000] = 0

                # Processing and storage of predictive output data
                dtf_predict_output_year_ages_heating.append(dtf_predict_output_year_age_heating.apply(lambda x: x.sum(),axis = 0).iloc[0])
                dtf_predict_output_year_ages_heating.append(year_observe)
                dtf_predict_output_year_ages_heating.append(vintage)
                dtf_predict_output_years_ages_heating.append(dtf_predict_output_year_ages_heating)
                dtf_predict_output_year_ages_cooling.append(dtf_predict_output_year_age_cooling.apply(lambda x: x.sum(), axis = 0).iloc[0])
                dtf_predict_output_year_ages_cooling.append(year_observe)
                dtf_predict_output_year_ages_cooling.append(vintage)
                dtf_predict_output_years_ages_cooling.append(dtf_predict_output_year_ages_cooling)
        dtf_predict_output_years_ages_heating = pd.DataFrame(dtf_predict_output_years_ages_heating)
        dtf_predict_output_years_ages_heating.to_excel('./energy/{city_code}/{city_code}-{e_path}-{w_scenario}-heating-tranist.xls'.format(city_code = city_code, e_path = e_path, w_scenario = w_scenario))
        dtf_predict_output_years_ages_cooling = pd.DataFrame(dtf_predict_output_years_ages_cooling)
        dtf_predict_output_years_ages_cooling.to_excel('./energy/{city_code}/{city_code}-{e_path}-{w_scenario}-cooling-tranist.xls'.format(city_code = city_code, e_path = e_path, w_scenario = w_scenario))


# The building energy consumption and building energy intensity of each year after prediction were calculated
def Predict_output_data_calculate(city_code, e_path, w_scenario, city_building_hvac, city_building_ages):
    # This has to be written row by column in excel
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('My Worksheet')
    # Read the transfer data, is a summary of the data.
    path_tranist = './energy/{city_code}/{city_code}-{e_path}-{w_scenario}-{city_building_hvac}-tranist.xls'.format(city_code = city_code, e_path = e_path, w_scenario = w_scenario, city_building_hvac = city_building_hvac)
    dtf_predict_tranist = pd.read_excel(path_tranist)

    # The predicted annual building energy intensity is arranged in the previous building stock format
    row = 0
    for i in range(len(city_building_ages)):
         # Building energy consumption for all observed years during the construction period
        energy_intensity_years_age = dtf_predict_tranist[dtf_predict_tranist[2] == city_building_ages[i]][0].to_list()
        # The building energy intensity of each year is written according to the situation of different built ages
        if i < len(city_building_ages) - 1:
            for j in range(city_building_ages[i + 1]-city_building_ages[i]):
                for col in range(81):
                    worksheet.write(row, col, label = energy_intensity_years_age[col])
                row = row + 1
        elif i == len(city_building_ages) - 1 and city_building_ages[-1] <= 2020:
            for j in range(2021-city_building_ages[-1]):
                for col in range(81):
                    worksheet.write(row, col, label = energy_intensity_years_age[col])
                row = row + 1
            # The starting point for the flow of new construction after 2020
            new_building = 1
            for j in range(2101 - 2021):
                for col in range(new_building, 81):
                    worksheet.write(row, col, label = energy_intensity_years_age[col])
                new_building = new_building + 1
                row = row + 1

        # You have to save it before you can read all the data
        workbook.save('./energy/{city_code}/{city_code}-{Envelope_Path}-{Weather_Scenario}-{city_building_hvac}-stock.xls'.format(city_code = city_code, Envelope_Path = e_path, Weather_Scenario = w_scenario, city_building_hvac = city_building_hvac))
    dtf_intensity_years_ages = pd.read_excel('./energy/{city_code}/{city_code}-{Envelope_Path}-{Weather_Scenario}-{city_building_hvac}-stock.xls'.format(city_code = city_code, Envelope_Path = e_path, Weather_Scenario = w_scenario, city_building_hvac = city_building_hvac))
    dtf_building_years_ages = pd.read_csv('./building_stock/n{city_code}_dynamic_stock.csv'.format(city_code = city_code))
    dtf_intensity_years_ages.columns = year_observes
    dtf_building_years_ages = dtf_building_years_ages.drop('age', axis = 1)
    dtf_building_years_ages.columns = year_observes

    # First, the energy consumption of all built ages in each year is calculated, and then the total energy consumption of each year is summed up, and then the total area of each year is divided to obtain the energy intensity of each year.
    dtf_energy_years_ages = dtf_intensity_years_ages * dtf_building_years_ages
    energy_year = dtf_energy_years_ages.apply(lambda x: x.sum(), axis = 0)
    area_year = dtf_building_years_ages.apply(lambda x: x.sum(), axis = 0)
    intensity_year = energy_year / area_year

    return energy_year, intensity_year


# All the functions related to the prediction are integrated, leaving only one parameter, the city number
def Predict(city_number):

    my_predict = Predict_surrogate_model()

    # Consistent data and data preprocessing are used for prediction and training
    dtf_train_input_heating, dtf_train_input_cooling, scalerX_heating, scalerY_heating, scalerX_cooling, scalerY_cooling = Train_input_data_process()

    # Read the information for each city from the master table
    city_code = city_information_table.iloc[city_number - 1, 0]
    city_climate_station = city_information_table.iloc[city_number - 1, 1]
    city_climate_zone = city_information_table.iloc[city_number - 1, 4]
    city_building_stories = city_information_table.iloc[city_number - 1, 7]
    city_building_ages_total = city_information_table.iloc[city_number - 1, 8 : 22].tolist()
    city_building_hvac_index = city_information_table.iloc[city_number - 1, 22]
    city_building_hvac_time_heating = city_information_table.iloc[city_number - 1, 23 : 25].tolist()
    city_building_hvac_time_cooling = city_information_table.iloc[city_number - 1, 25 : 27].tolist()

    print(city_code)
    if city_climate_zone == 'warm':
        return

    # According to the information of each city to synthesize the required forecast input data, output and collate the forecast results
    energy_years = []
    intensity_years = []
    ouput_scenario_indexs = []

    # Determine whether it has been predicted by the model, or simply by replicating previously predicted results
    is_predict = 0

    # Counties with the same climate and building information directly copy the predicted results
    for city_number_from_0 in range(city_number):

        # Check it backwards, because the same information is usually near to each other
        city_number_predicted = city_number - 1 - city_number_from_0

        # Judging from three dimensions: city_climate_station、 city_climate_zone、 city_building_stories
        if city_number_predicted < city_number - 1  and city_climate_station == city_information_table.iloc[city_number_predicted  , 1] \
                                and city_climate_zone == city_information_table.iloc[city_number_predicted , 4]\
                                and city_building_stories == city_information_table.iloc[city_number_predicted  , 7]:
            print('copy!')
            city_code_predicted = city_information_table.iloc[city_number_predicted , 0]
            if int(city_building_hvac_index) == 2:
                city_building_hvacs = ['heating', 'cooling']
            else:
                city_building_hvacs = [['heating', 'cooling'][int(city_building_hvac_index)]]
            my_predict.Copy_predicted(city_code_predicted, city_code, city_building_hvacs)

            # after copy, is_predict = 1 , no need to predict, and break
            is_predict = 1
            break

    # Counties that do not have the same information use the surrogate model for the first forecast
    if is_predict == 0:
        for i in range(len(Envelope_Paths)):

            # Different envelope paths correspond to several building ages
            city_building_ages = city_building_ages_total[[0, 3, 6][i] : [3, 6, 14][i]]
            city_building_ages = [int(i) for i in city_building_ages]
            while 0 in city_building_ages:
                city_building_ages.remove(0)

            for j in range(len(Weather_Scenarios)):
                print(Envelope_Paths[i] + '-' + Weather_Scenarios[j])
                if int(city_building_hvac_index) == 2:
                    city_building_hvacs = ['heating', 'cooling']
                    city_building_hvac_time = city_building_hvac_time_heating + city_building_hvac_time_cooling
                    my_predict.Multiple_predict(city_code, Envelope_Paths[i], Weather_Scenarios[j], city_climate_station, city_building_stories, city_climate_zone, city_building_ages, city_building_hvac_time, scalerX_heating, scalerY_heating, scalerX_cooling, scalerY_cooling)
                else:
                    city_building_hvacs = [['heating', 'cooling'][int(city_building_hvac_index)]]
                    city_building_hvac_time = [city_building_hvac_time_heating, city_building_hvac_time_cooling][int(city_building_hvac_index)]
                    my_predict.Single_predict(city_code, Envelope_Paths[i], Weather_Scenarios[j], city_climate_station, city_building_stories, city_climate_zone, city_building_ages, city_building_hvacs.iloc[0], city_building_hvac_time, scalerX_heating, scalerY_heating, scalerX_cooling, scalerY_cooling)

    # The prediction results are calculated by scenarios
    for i in range(len(Envelope_Paths)):
        city_building_ages = city_building_ages_total[[0, 3, 6][i]: [3, 6, 14][i]]
        city_building_ages = [int(i) for i in city_building_ages]
        while 0 in city_building_ages:
            city_building_ages.remove(0)
        for j in range(len(Weather_Scenarios)):
            for city_building_hvac in city_building_hvacs:
                energy_year, intensity_year = Predict_output_data_calculate(city_code, Envelope_Paths[i], Weather_Scenarios[j], city_building_hvac, city_building_ages)
                ouput_scenario_indexs.append(city_building_hvac + '-' + Envelope_Paths[i] + '-' + Weather_Scenarios[j])
                energy_years.append(energy_year)
                intensity_years.append(intensity_year)

    # The output is the result of time series under nine scenarios
    intensity_years = pd.DataFrame(intensity_years)
    intensity_years.index = ouput_scenario_indexs
    energy_years = pd.DataFrame(energy_years)
    energy_years.index = ouput_scenario_indexs
    energy_years.to_csv('./energy/{city_code}/{city_code}-energys.csv'.format(city_code = city_code))
    intensity_years.to_csv('./energy/{city_code}/{city_code}-intensitys.csv'.format(city_code = city_code))



# Record start time
start = time.time()

# Sample(10000)
# Call scope, which must start at 1
'''
for sample_order in range(1,2): # Range of all samples, a total of 5000 samples must start at 1
    print(sample_order)
    Simulate(sample_order)
'''

# Train('heating')

'''
for n in range(1, 1678):
    Predict(n) # Add a time sleep
    time.sleep(0)
'''

# Record end time
end = time.time()

# Print elapsed time
print('Time taken:', end - start)