import numpy as np

original_list = [18, 0, 0, 0, 313, [24, 24, 24, 24, 24, 24, 24, 24, 22, 96]]

# Convert the list and nested list to NumPy arrays
array = np.array(original_list, dtype=object)

# Flatten the array
flat_array = np.hstack(array)

# Print the flat array
print(flat_array)