# Project 1 - Visible Numbers

# Pseudocode
    # Important notes: 
        # The matrix size will only be between 2 and 4
        # Each row and column must contain unique numbers, meaning no duplicates in any row or column (latin square), this can be checked

    # Steps
    #   1. Read Input
    #   2. Initialize empty grid filled with 0s
    #   3. Begin DFS at top left position
    #   4. Try "valid" numbers based on visible numbers 
    #       - Will need a function that checks what numbers "could" be in that spot
    #   5. If valid, move to next spot and repeat (recurisvely call DFS)
    #   6. If invalid, set number back to 0 and iterate to next valid number

def parseInput():
    # n = 4
    # top = 4, 3, 2, 1
    # bottom = 1, 2, 2, 2
    # left = 4, 3, 2, 1
    # right = 1, 2, 2, 2

    # n = 2
    # top = 1, 2
    # bottom = 2, 1
    # left = 1, 2
    # right = 2, 1
    
    n = int(input())
    top = list(map(int, input().split()))
    bottom = list(map(int, input().split()))
    left = list(map(int, input().split()))
    right = list(map(int, input().split()))
    
    return n, top, bottom, left, right

# Check if a row/column satsfies the visibility constraints from both ends. You can see a person if they're taller than all 
# people in front of them and count how many people are visible from each direction. 
def checkVisibility(line, startVisibility, endVisibility):
    # If line contains 0 it's incomplete, we can't evaluate visibility yet
    # Return True to allow continued exploration
    if 0 in line:
        return True
    
    # Variables to track the tallest person seen so far from each direction
    heightMaxStart = 0      # Tallest person seen from start
    visibleFromStart = 0    # Count of visible people from start
    heightMaxEnd = 0        # Tallest person seen from end  
    visibleFromEnd = 0      # Count of visible people from end

    # Scan from both directions simultaneously
    for i in range(len(line)):
        # (left->right rows, top->bottom columns)
        if line[i] > heightMaxStart:
            visibleFromStart += 1       # This person is visible
            heightMaxStart = line[i]    # Update tallest seen

        # (right->left rows, bottom->top columns)
        if line[-(i + 1)] > heightMaxEnd:
            visibleFromEnd += 1                 # This person is visible
            heightMaxEnd = line[-(i + 1)]       # Update tallest seen

    # Check if actual visibility counts match input 
    return visibleFromStart == startVisibility and visibleFromEnd == endVisibility

# Validates that the current partial solution doesn't violate any constraints.
# This is called after placing a number at position (row, col). Returns true if a partial solution is valid. 
# False if constraints are violated (triggers backtracking).
def isValidPartialSolution(matrix, row, col, matrixSize, top, bottom, left, right):
    # Check #1 No duplicates in current row (up to current column)
    currentRow = matrix[row][:col+1]  # Get row from start to current position

    # Count unique numbers and compare to total 
    uniqueInRow = len(set(num for num in currentRow if num != 0))
    totalInRow = len([num for num in currentRow if num != 0])
    if uniqueInRow != totalInRow:
        return False  # Found duplcate in row
    
    # Check #2 No duplicates in current column (only up to current row)
    currentCol = [matrix[r][col] for r in range(row+1)]  # Get column from top to current position
    # Count unique numbers and compare to total numbers again
    uniqueInCol = len(set(num for num in currentCol if num != 0))
    totalInCol = len([num for num in currentCol if num != 0])
    if uniqueInCol != totalInCol:
        return False  # Found duplicate in column
    
    # Check #3 If current row is complete, validate its visibility constraints
    if col == matrixSize - 1 and 0 not in matrix[row]:
        # Row is complete, check left and right visibility clues
        if not checkVisibility(matrix[row], left[row], right[row]):
            return False  # Row violates visibility constraints
    
    # Check #4 If current column is complete, validate its visibility constraints  
    if row == matrixSize - 1:
        # We're in the last row, check if this column is complete
        column = [matrix[r][col] for r in range(matrixSize)]
        if 0 not in column:  # Column is complete
            # Check top and bottom visibility clues
            if not checkVisibility(column, top[col], bottom[col]):
                return False  # Column violates visibility constraints
    
    # All checks passed, partial solution is valid
    return True

# Get all numbers 1 to N that can be legally placed at position row, col. Returns a list of said valid numbers.
def getValidNumbers(matrix, row, col, matrixSize):
    # Get all numbers already used in this row, excluding 0 
    usedInRow = set(matrix[row])
    
    # Get all numbers already used in this column, excluding 0 
    usedInCol = set(matrix[r][col] for r in range(matrixSize))
    
    # Combine both sets to get all forbidden numbers
    used = usedInRow.union(usedInCol)
    used.discard(0)  # Remove 0 since it represents empty cells, not a forbidden number
    
    # Return all numbers frm 1 to N that aren't forbidden
    return [num for num in range(1, matrixSize + 1) if num not in used]

# Depth first with backtracking, fill grid cell by cell, at each cell try all valid numbers. If something doesn't fit, backtrack and try the next number. 
# If all the numbers fail we backtrack to the previous cell. Continues until complete or all possibilties are exhausted. 
def DFSSolve(matrixSize, top, bottom, left, right):
    def DFSRecursive(matrix, row, col):
        # If we've moved past the last row, puzzle is solved
        if row == matrixSize:
            return True
        
        # Calculate next position to fill
        # Move right in current row, or if it's at the end move to start
        nextRow, nextCol = (row, col + 1) if col < matrixSize - 1 else (row + 1, 0)
        
        # Get all numbers that can legally be placed at current position
        validNumbers = getValidNumbers(matrix, row, col, matrixSize)
        
        # Try each valid number
        for num in validNumbers:
            # Put specific number in current cell
            matrix[row][col] = num
            
            # Check if this placement violates any constraints
            if isValidPartialSolution(matrix, row, col, matrixSize, top, bottom, left, right):
                # The placement is valid so we try to complete the rest of the puzzle
                if DFSRecursive(matrix, nextRow, nextCol):
                    return True  # Solution found!
            
            # If the function hasn't returned true at this point the individual num doesn't work, remove it and try next
            matrix[row][col] = 0
        
        # No valid number worked at this position so we backtrack further
        return False
    
    # ------------ This is where DFSSolve actually starts ------------
    # Initialize empty grid (0 = unfilled cell)
    matrix = [[0 for _ in range(matrixSize)] for _ in range(matrixSize)]
    
    # Start DFS from top-left corner (0, 0)
    if DFSRecursive(matrix, 0, 0):
        return matrix    # Solution found
    else:
        return None      # No solution exists

def main():    
    # Read matrix parameters from input
    matrixSize, top, bottom, left, right = parseInput()
    
    # Solve the puzzle using DFS
    solution = DFSSolve(matrixSize, top, bottom, left, right)
    
    # Output the solution
    if solution is not None:
        # Print each row of the solution matrix
        for row in solution:
            print(" ".join(map(str, row)))
    # If no solution exists there'll be no output

# run main function when script is executed
if __name__ == "__main__":
    main()