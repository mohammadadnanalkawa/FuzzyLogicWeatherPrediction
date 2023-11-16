# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 08:09:25 2019

@author: ADnAN
"""
from keras.models import Sequential
from keras.layers import Dense
import numpy
import csv

# fix random seed for reproducibility
numpy.random.seed(7)

def predict(FILE_NAME ,THE_SEPERATOR_ROW, PARAMETERS_STARTING_COLUMS, RESULTS_STARTING_COLUM, RESULTS_ENDING_COLUM):
    
    # load pima indians dataset
    dataset = numpy.loadtxt(FILE_NAME, delimiter=";")

    
    # split into input (X) and output (Y) variables
    training_parameters = dataset[:THE_SEPERATOR_ROW,PARAMETERS_STARTING_COLUMS:RESULTS_STARTING_COLUM]
    training_result = dataset[:THE_SEPERATOR_ROW,RESULTS_STARTING_COLUM:]
    
    testing_parameters = dataset[THE_SEPERATOR_ROW:,PARAMETERS_STARTING_COLUMS:RESULTS_STARTING_COLUM]
    testing_result = dataset[THE_SEPERATOR_ROW:,RESULTS_STARTING_COLUM:]
    
    
    
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=(RESULTS_STARTING_COLUM - PARAMETERS_STARTING_COLUMS ), activation='sigmoid'))
    model.add(Dense(8, activation='sigmoid'))
    model.add(Dense((RESULTS_ENDING_COLUM - RESULTS_STARTING_COLUM), activation='sigmoid'))
    
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    	
    # Fit the model
    model.fit(training_parameters, training_result, epochs=55, batch_size=10)
    
    
    # evaluate the model
    scores = model.evaluate(testing_parameters, testing_result)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    predictions = model.predict(testing_parameters)
    # round predictions
    #predicted_result = [round(x[0]) for x in predictions]
    
    # writing the results into an output file
    csvfile = FILE_NAME+"_output.csv"
    
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n', delimiter=";")
        writer.writerows(predictions) 
    
    # compare predicteed results with the given results
    
predict("DataSet\Temp-2-mid-hi.csv",3226,3,5,7)
predict("DataSet\Wind-5.csv",3226,3,5,10)
predict("DataSet\Rain-5.csv",3226,3,5,10)
predict("DataSet\Humd-4.csv",3226,3,5,9)