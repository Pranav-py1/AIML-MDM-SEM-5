# AI/ML Lab 

## What is this?

This folder contains two small AI search programs written in Python:

- `8puzzle.py` -> solves the 8-puzzle using different search strategies
- `Nqueens.py` -> solves the N-Queens problem using local search and CSP backtracking

The programs are kept pretty simple and menu-driven, so they can be run directly from the terminal.

---

# 1. 8-Puzzle Search

The 8-puzzle is basically a 3x3 board with tiles numbered from 1 to 8 and one blank space (`0`).

The goal is:

```text
1 2 3
4 5 6
7 8 0
```

The program lets us try different search algorithms on the same starting state.

### Algorithms used

1. BFS
2. DFS
3. UCS
4. Greedy
5. A*

For UCS, Greedy and A*, the program uses the Manhattan distance heuristic where required. The code also checks whether the given puzzle state is solvable before starting the search. fileciteturn0file0L28-L49

### How to run

```bash
python 8puzzle.py
```

Then enter the 9 tiles separated by spaces.

Example:

```text
4 1 3 2 0 6 7 5 8
```

Here `0` represents the blank space.

After that, select an option:

```text
1.BFS
2.DFS
3.UCS
4.GREEDY
5.A*
6.RUN ALL
```

Choosing `6` runs all five algorithms one after another. fileciteturn0file0L102-L115

### Example output

```text
start: (4, 1, 3, 2, 0, 6, 7, 5, 8)
goal: (1, 2, 3, 4, 5, 6, 7, 8, 0)

BFS      path=6      exp=102    mf=65     t=0.00097s
DFS      path=926    exp=953    mf=745    t=0.00835s
UCS      path=6      exp=102    mf=65     t=0.00101s
GREEDY   path=6      exp=8      mf=10     t=0.00012s
A*       path=6      exp=8      mf=10     t=0.00012s
```

The exact numbers can change depending on the input and execution.

### What the output means

- `path` -> number of moves needed to reach the goal
- `exp` -> number of states expanded
- `mf` -> maximum frontier size
- `t` -> execution time in seconds

The program stores parent nodes and reconstructs the path once the goal is found. fileciteturn0file0L51-L58

---

# 2. N-Queens

The N-Queens problem is about placing `N` queens on an `N x N` chessboard such that no two queens attack each other.

The program provides two approaches:

1. Hill Climbing
2. Backtracking as a CSP

There is also an option to run both.

### How to run

```bash
python Nqueens.py
```

Enter the value of `N`.

For example:

```text
enter N  8
```

Then select:

```text
1.HILL CLIMBING(local search)
2.BACKTRACKING(CSP)
3.RUN BOTH
```

Choosing `3` runs both approaches. fileciteturn0file1L69-L84

---

## Hill Climbing

The hill climbing approach starts with a random board configuration and tries to reduce the number of conflicts.

The conflict function checks:

- queens in the same column
- queens on the same diagonal

If the conflict count becomes `0`, a solution is found. The program can restart with a new random configuration if it gets stuck. fileciteturn0file1L3-L10 fileciteturn0file1L12-L38

Example:

```text
HILLCLIMB restarts=11 steps=46 t=0.01324s solved=True
```

---

## Backtracking

The backtracking approach builds the solution row by row.

Before placing a queen, it checks whether the position is safe by making sure that:

- no queen is already in the same column
- no queen is on the same diagonal

If a position does not work, the program goes back and tries another position. fileciteturn0file1L40-L63

Example:

```text
BACKTRACK nodes=876 t=0.00039s solved=True
```

---

# Files

```text
8puzzle.py
Nqueens.py
README.md
```

## Requirements

Python 3.x

Both programs only use Python's standard library, so there is no separate package installation required.

## In short

This is basically a small collection of AI search implementations for the lab:

```text
8-Puzzle
   |
   +-- BFS
   +-- DFS
   +-- UCS
   +-- Greedy
   +-- A*

N-Queens
   |
   +-- Hill Climbing
   +-- Backtracking (CSP)
```

The main idea is to run the same problem using different search methods and compare how they behave in terms of solution path, states explored, frontier size, restarts/steps, and execution time.
