assignments = []

def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """

    # Don't waste memory appending actions that don't actually change any values
    if values[box] == value:
        return values

    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values


def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """

    # Find all instances of naked twins
    # Eliminate the naked twins as possibilities for their peers
    for unit in unitlist:
        value_to_box = {}
        for box in unit:
            if not values[box] in value_to_box:
                value_to_box[values[box]] = [box]
            else:
                value_to_box[values[box]].append(box)
        for value in value_to_box.keys():
            if len(value)>1 and len(value) == 2 and len(value_to_box[value]) == 2:
                #print("Found naked twins in unit: \n", unit)
                #print("naked twin boxes {} have the same value {} ".format(value_to_box[value], value))
                for v in value:
                    for box in (set(unit)-set(value_to_box[value])):
                        if v in values[box]:
                            #print("update box {} by eliminating value {} ".format(box,v))
                            assign_value(values, box, values[box].replace(v, ''))
    return values


def cross(A,B):
    "Cross product of elements in A and elements in B."
    return [a+b for a in A for b in B]


def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    chars = []
    digits = '123456789'
    for c in grid:
        if c in digits:
            chars.append(c)
        if c == '.':
            chars.append(digits)
    assert len(chars) == 81
    return dict(zip(boxes, chars))


def display(values):
    """
    Display the values as a 2-D grid.
    Args:
        values(dict): The sudoku in dictionary form
    """
    width = 1 + max(len(values[s]) for s in values.keys())
    line = '+'.join(['-' * (width * 3)] * 3)
    for r in rows:
        print(''.join(values[r + c].center(width) + ('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': 
            print(line)


def eliminate(values):
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    i = 0
    for box in solved_values:
        digit = values[box]
        for peer in peers[box]:
            i +=1
            assign_value(values, peer, values[peer].replace(digit, ''))
    return values


def only_choice(values):
    for unit in unitlist:
        for digit in '123456789':
            dplaces = [box for box in unit if digit in values[box]]
            if len(dplaces) == 1:
                assign_value(values, dplaces[0], digit)
    return values


def reduce_puzzle(values):
    stalled = False
    while not stalled:
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        values = eliminate(values)
        values = only_choice(values)
        values = naked_twins(values)
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        stalled = solved_values_before == solved_values_after
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values


def search(values):
    if not reduce_puzzle(values):
        return False

    # check if puzzle is solved
    if all(len(values[s]) == 1 for s in boxes):
            return values

    # find the unfilled box with the fewest choice possibilities
    box = min([box for box in boxes if len(values[box])>1], key=lambda x: len(x))
    for value in values[box]:
        # create a new sudoku and update its values
        new_sudoku = values.copy()
        new_sudoku[box] = value
        attemp = search(new_sudoku)
        if attemp:
            return attemp


def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    values = grid_values(grid)
    search(values)
    return assignments[-1]


rows = 'ABCDEFGHI'
cols = '123456789'

boxes = cross(rows, cols)
row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')]
diagonal_units = [[rows[i]+cols[i] for i in range(len(rows))],
                 [rows[i]+cols[len(rows)-i-1] for i in range(len(rows))]]

unitlist = row_units + column_units + square_units + diagonal_units
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s], [])) - set([s])) for s in boxes)


if __name__ == '__main__':
    #diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    diag_sudoku_grid = "9.1....8.8.5.7..4.2.4....6...7......5..............83.3..6......9................"
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
