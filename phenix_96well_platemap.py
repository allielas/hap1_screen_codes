# %%
import math
import os
import pandas as pd
import numpy as np

plate_path = 'plate_metadata/' #'/home/mattiazzilab/Documents/Allie_Scripts/May 8 seeding.csv'
export_path = 'plate_metadata/'#'/mnt/bigdisk1/Allie_S/Replicative_Age_Project/Data Mining/metadata/'


# %%
#calculate passage number when I ran out of passage 5 vials ;;
def passage_number(time):
    base_vial = 6
    if time > 2:
        base_vial = base_vial-1 
    pnum = base_vial + (3 * time)
    return pnum


def well_namer(row, col):
    well_name = str(chr(ord('@')+ row)) + str(col).rjust(2, '0')  #make the number have a left align, adding a zero
    return well_name

import re

def extract_number(string):
    match = re.search(r'\d+', string)
    if match:
        return int(match.group())
    else:
        return None
# %%
#Get 96-well plate CSV file from benchiling/Notion NOTE: don't initialze an empty df, use a list and convert after.
def load_plate_df(path):    
    raw_plate_df = pd.read_csv(path, header=0, usecols=range(1,13)).dropna(axis=1,how='all')
    plate_df = raw_plate_df.dropna(axis=0, thresh=2)
    print(plate_df.shape) #Checks if correct number of rows and columns
    
    return plate_df

def make_map_df():
    '''
    Return a df with the required columns
    '''
    plate_map_df = pd.DataFrame(columns = ['Metadata_Well', 'Metadata_WellRow', 'Metadata_WellColumn', 'Metadata_Field',
                                'TimepointName', 'SerialPassage_BatchNumber', 'AgeGroup', 'PassageNumber', 'Staining', 'Drug'])
    print(plate_map_df.columns)
    return plate_map_df
#plate_df.head(13)
#print(columns_row)

# %%
def export_platemap_csv(plate_df, export_path):
    plate_map_df = make_map_df()
    columns_row = plate_df.columns#get_columns_row(plate_df)
    for index,data in plate_df.iterrows():
        row = data.to_list()
        for count,value in enumerate(row):
            curr_well = value
            if pd.isna(curr_well):
                continue
            row_list = []
            # get string data and label of row in df (e.g. col1, text= R1T0 EAA1-488 Tfn-647)
            # regex to separate into different variables
            # then add them to dict with their respective col index (label) and index of the row in the column (column.index)
            for i in range(40):
                
                row_entry = {}
                
                row_index = index+1  #Make it 1-indexed
                
                column_index = columns_row[count]  # Use header row for column index
                #print(column_index)
            
                if ~np.isnan(row_index): 
                    well_name = well_namer(row_index, column_index)
                else:
                    well_name = 'Empty'
                    continue
                
                #seperate well metadata by space
                text = curr_well.split(' ')
                
                #use regex to extract the numerical bits
                serial_passage_batch = extract_number(text[0])
                flagged_batch = "Flagged" in text[0] #flag passage if we have "Flagged" in the serial passage batch
                age_group = extract_number(text[1]) #time used to mean age group - deprecated term but still used in code
                passage_num = extract_number(text[2]) #passage_number(Int(time)) - use the function if you don;t have passage number in the table
                
                #grab drug, name and stains
                if '_' in text[1]: #if there is an underscore than the well is drug-treated for this group
                    drug = text[1].split('_')[1] #grab the drug name past the underscore
                else:
                    drug = 'None'
                
                name = text[0] + ' ' + text[1] + ' ' + text[2]
                stains = ' '.join(text[3:len(text)]) #Use if 3 passage number is in the table values
                
                field = i+1 #information for the field is just 1-40, nothing else changes

                print(name, serial_passage_batch, age_group, passage_num, field, stains, drug, sep=',') 
        
                row_entry.update({'Metadata_Well': well_name, 'Metadata_WellRow': row_index, 'Metadata_WellColumn': column_index, 'Metadata_Field': field,
                                'TimepointName':name, 'SerialPassage_BatchNumber': serial_passage_batch, 'AgeGroup': age_group, 'PassageNumber': passage_num,
                                'Staining': stains, 'Drug': drug, 'FlaggedBatch': flagged_batch})

                row_list.append(row_entry)
                
            rows = pd.DataFrame(row_list)
            plate_map_df = pd.concat([plate_map_df, rows],ignore_index=True)
            #column_index = column_index+1
    plate_map_df.to_csv(os.path.join(export_path,'map.csv'), index=False)
    

# %%
for root, dirs, files in os.walk(plate_path):
    for filename in files:
        if filename.endswith(".csv") and "map" not in filename:
            file_path = os.path.join(root,filename)
            plate_df = load_plate_df(os.path.abspath(file_path))
            #display(plate_df)
            export_path = os.path.abspath(root)
            print(f"Exporting {filename} to {export_path}")
            export_platemap_csv(plate_df, export_path)
            

# %%



