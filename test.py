import numpy as np
import pandas as pd

def process_set(set):
    # Removing the $ and , characters and converting to integers
    set = np.array([int(s.replace('$', '').replace(',', '')) for s in set])

    # Calculating the mean, median, and standard deviation for each set and rounding to 2 decimal places
    mean_set = round(np.mean(set), 2)
    median_set = round(np.median(set), 2)
    std_set = round(np.std(set), 2)

    return mean_set, median_set, std_set

# Defining the sets of numbers
set1 = np.array(['$51,300', '$54,600'])
set2 = np.array(['$59,200', '$62,500', '$65,900'])
set3 = np.array(['$70,500', '$73,900', '$77,900', '$82,100'])
set4 = np.array(['$86,900', '$95,500', '$106,500', '$119,800', '$132,700', '$143,800', '$154,200', '$165,100', '$181,800'])

# Process each set
mean_set1, median_set1, std_set1 = process_set(set1)
mean_set2, median_set2, std_set2 = process_set(set2)
mean_set3, median_set3, std_set3 = process_set(set3)
mean_set4, median_set4, std_set4 = process_set(set4)

# Create a dictionary with the calculated values
data = {
    'Junior Specialist': [mean_set1, median_set1, std_set1],
    'Assistant Specialist': [mean_set2, median_set2, std_set2],
    'Associate Specialist': [mean_set3, median_set3, std_set3],
    'Specialist': [mean_set4, median_set4, std_set4]
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data, index=['Mean', 'Median', 'Standard Deviation'])

print(df)