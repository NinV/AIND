# Artificial Intelligence Nanodegree
## Introductory Project: Diagonal Sudoku Solver

# Question 1 (Naked Twins)
Q: How do we use constraint propagation to solve the naked twins problem?  
A: We iterate through all unit in unit_list. For each unit, firstly we find the 2 naked twins boxes by counting the value frequency of all boxes in a unit. We create a value_to_box dictionay to map each value to the boxes associated with it  {value:[box1, box2, ...]}. Then we find which value has the length of 2 and has only 2 boxes associated with. Finaly, we eliminate all value of naked twins boxes in the unsolved box in the unit

# Question 2 (Diagonal Sudoku)
Q: How do we use constraint propagation to solve the diagonal sudoku problem?  
A: Firstly, we create a list of diagonal_units then merge it with the unit_list of normal sudoku. The unit_list of diagonal sudoku now contains row_units, column_units, square_units and diagonal_units. Now we can apply constraint propagation to the new unit_list to solve the diagonal sudoku problem

