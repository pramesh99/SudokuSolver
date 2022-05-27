def getel(s):
    """Returns the unique element in a singleton set (or list)."""
    assert len(s) == 1
    return list(s)[0]

import json

class Sudoku(object):
    def __init__(self, elements):
        """Elements can be one of:
        Case 1: a list of 9 strings of length 9 each.
        Each string represents a row of the initial Sudoku puzzle,
        with either a digit 1..9 in it, or with a blank or _ to signify
        a blank cell.
        Case 2: an instance of Sudoku.  In that case, we initialize an
        object to be equal (a copy) of the one in elements.
        Case 3: a list of list of sets, used to initialize the problem."""
        if isinstance(elements, Sudoku):
            # We let self.m consist of copies of each set in elements.m
            self.m = [[x.copy() for x in row] for row in elements.m]
        else:
            assert len(elements) == 9
            for s in elements:
                assert len(s) == 9
            # We let self.m be our Sudoku problem, a 9x9 matrix of sets.
            self.m = []
            for s in elements:
                row = []
                for c in s:
                    if isinstance(c, str):
                        if c.isdigit():
                            row.append({int(c)})
                        else:
                            row.append({1, 2, 3, 4, 5, 6, 7, 8, 9})
                    else:
                        assert isinstance(c, set)
                        row.append(c)
                self.m.append(row)


    def show(self, details=False):
        """Prints out the Sudoku matrix.  If details=False, we print out
        the digits only for cells that have singleton sets (where only
        one digit can fit).  If details=True, for each cell, we display the
        sets associated with the cell."""
        if details:
            print("+-----------------------------+-----------------------------+-----------------------------+")
            for i in range(9):
                r = '|'
                for j in range(9):
                    # We represent the set {2, 3, 5} via _23_5____
                    s = ''
                    for k in range(1, 10):
                        s += str(k) if k in self.m[i][j] else '_'
                    r += s
                    r += '|' if (j + 1) % 3 == 0 else ' '
                print(r)
                if (i + 1) % 3 == 0:
                    print("+-----------------------------+-----------------------------+-----------------------------+")
        else:
            print("+---+---+---+")
            for i in range(9):
                r = '|'
                for j in range(9):
                    if len(self.m[i][j]) == 1:
                        r += str(getel(self.m[i][j]))
                    else:
                        r += "."
                    if (j + 1) % 3 == 0:
                        r += "|"
                print(r)
                if (i + 1) % 3 == 0:
                    print("+---+---+---+")


    def to_string(self):
        """This method is useful for producing a representation that
        can be used in testing."""
        as_lists = [[list(self.m[i][j]) for j in range(9)] for i in range(9)]
        return json.dumps(as_lists)


    @staticmethod
    def from_string(s):
        """Inverse of above."""
        as_lists = json.loads(s)
        as_sets = [[set(el) for el in row] for row in as_lists]
        return Sudoku(as_sets)


    def __eq__(self, other):
        """Useful for testing."""
        return self.m == other.m

"""Let us input a problem (the Sudoku example found on [this Wikipedia page](https://en.wikipedia.org/wiki/Sudoku)) and check that our serialization and deserialization works."""

# Let us ensure that nose is installed.
try:
    from nose.tools import assert_equal, assert_true
    from nose.tools import assert_false, assert_almost_equal
except:
    !pip install nose
    from nose.tools import assert_equal, assert_true
    from nose.tools import assert_false, assert_almost_equal

from nose.tools import assert_equal

sd = Sudoku([
    '53__7____',
    '6__195___',
    '_98____6_',
    '8___6___3',
    '4__8_3__1',
    '7___2___6',
    '_6____28_',
    '___419__5',
    '____8__79'
])
sd.show()
sd.show(details=True)
s = sd.to_string()
sdd = Sudoku.from_string(s)
sdd.show(details=True)
assert_equal(sd, sdd)

"""Let's test the constructor statement when passed a Sudoku instance."""

sd1 = Sudoku(sd)
assert_equal(sd, sd1)

"""## Constraint propagation

When the set in a Sudoku cell contains only one element, this means that the digit at that cell is known. 
We can then propagate the knowledge, ruling out that digit in the same row, in the same column, and in the same 3x3 cell. 

We first write a method that propagates the constraint from a single cell.  The method will return the list of newly-determined cells, that is, the list of cells who also now (but not before) are associated with a 1-element set.  This is useful, because we can then propagate the constraints from those cells in turn.  Further, if an empty set is ever generated, we raise the exception Unsolvable: this means that there is no solution to the proposed Sudoku puzzle. 

## Propagating a single cell
"""

class Unsolvable(Exception):
    pass


def sudoku_ruleout(self, i, j, x):
    """The input consists in a cell (i, j), and a value x.
    The function removes x from the set self.m[i][j] at the cell, if present, and:
    - if the result is empty, raises Unsolvable;
    - if the cell used to be a non-singleton cell and is now a singleton
      cell, then returns the set {(i, j)};
    - otherwise, returns the empty set."""
    c = self.m[i][j]
    n = len(c)
    c.discard(x)
    self.m[i][j] = c
    if len(c) == 0:
        raise Unsolvable()
    return {(i, j)} if 1 == len(c) < n else set()

Sudoku.ruleout = sudoku_ruleout

"""The method `propagate_cell(ij)` takes as input a pair `ij` of coordinates.  If the set of possible digits `self.m[i][j]` for cell i,j contains more than one digit, then no propagation is done.  If the set contains a single digit `x`, then we: 

* Remove `x` from the sets of all other cells on the same row, column, and 3x3 block. 
* Collect all the newly singleton cells that are formed, due to the digit `x` being removed, and we return them as a set. 

Complete the implementation to take care of the column and 3x3 block as well.
"""

### Define cell propagation

def sudoku_propagate_cell(self, ij):
    """Propagates the singleton value at cell (i, j), returning the list
    of newly-singleton cells."""
    i, j = ij
    if len(self.m[i][j]) > 1:
        # Nothing to propagate from cell (i,j).
        return set()
    # We keep track of the newly-singleton cells.
    newly_singleton = set()
    x = getel(self.m[i][j]) # Value at (i, j).
    # Same row.
    for jj in range(9):
        if jj != j: # Do not propagate to the element itself.
            newly_singleton.update(self.ruleout(i, jj, x))
    # Same column.
    for ii in range(9):
        if ii != i:
            newly_singleton.update(self.ruleout(ii, j, x))
    # Same block of 3x3 cells.
    for ii in range(3):
        for jj in range(3):
            r = ii + 3 * (i // 3) 
            c = jj + 3 * (j // 3)
            if (r != i) and (c != j):
                newly_singleton.update(self.ruleout(r, c, x))

    # Returns the list of newly-singleton cells.
    return newly_singleton

Sudoku.propagate_cell = sudoku_propagate_cell

### Tests for cell propagation

tsd = Sudoku.from_string('[[[5], [3], [2], [6], [7], [8], [9], [1, 2, 4], [2]], [[6], [7], [1, 2, 4, 7], [1, 2, 3], [9], [5], [3], [1, 2, 4], [8]], [[1, 2], [9], [8], [3], [4], [1, 2], [5], [6], [7]], [[8], [5], [9], [1, 9, 7], [6], [1, 4, 9, 7], [4], [2], [3]], [[4], [2], [6], [8], [5], [3], [7], [9], [1]], [[7], [1], [3], [9], [2], [4], [8], [5], [6]], [[1, 9], [6], [1, 5, 9, 7], [9, 5, 7], [3], [9, 7], [2], [8], [4]], [[9, 2], [8], [9, 2, 7], [4], [1], [9, 2, 7], [6], [3], [5]], [[3], [4], [2, 3, 4, 5], [2, 5, 6], [8], [6], [1], [7], [9]]]')
tsd.show(details=True)
try:
    tsd.propagate_cell((0, 2))    
except Unsolvable:
    print("Good! It was unsolvable.")
else:
    raise Exception("Hey, it was unsolvable")

tsd = Sudoku.from_string('[[[5], [3], [2], [6], [7], [8], [9], [1, 2, 4], [2, 3]], [[6], [7], [1, 2, 4, 7], [1, 2, 3], [9], [5], [3], [1, 2, 4], [8]], [[1, 2], [9], [8], [3], [4], [1, 2], [5], [6], [7]], [[8], [5], [9], [1, 9, 7], [6], [1, 4, 9, 7], [4], [2], [3]], [[4], [2], [6], [8], [5], [3], [7], [9], [1]], [[7], [1], [3], [9], [2], [4], [8], [5], [6]], [[1, 9], [6], [1, 5, 9, 7], [9, 5, 7], [3], [9, 7], [2], [8], [4]], [[9, 2], [8], [9, 2, 7], [4], [1], [9, 2, 7], [6], [3], [5]], [[3], [4], [2, 3, 4, 5], [2, 5, 6], [8], [6], [1], [7], [9]]]')
tsd.show(details=True)
assert_equal(tsd.propagate_cell((0, 2)), {(0, 8), (2, 0)})

"""### Propagating all cells, once

The simplest thing we can do is propagate each cell, once. 
"""

def sudoku_propagate_all_cells_once(self):
    """This function propagates the constraints from all singletons."""
    for i in range(9):
        for j in range(9):
            self.propagate_cell((i, j))

Sudoku.propagate_all_cells_once = sudoku_propagate_all_cells_once

sd = Sudoku([
    '53__7____',
    '6__195___',
    '_98____6_',
    '8___6___3',
    '4__8_3__1',
    '7___2___6',
    '_6____28_',
    '___419__5',
    '____8__79'
])
sd.show()
sd.propagate_all_cells_once()
sd.show()
sd.show(details=True)

"""## Propagating all cells, repeatedly

As we propagate the constraints, cells that did not use to be singletons may have become singletons.  For eample, in the above example, the center cell has become known to be a 5: we need to make sure that also these singletons are propagated. 

This is why we have written propagate_cell so that it returns the set of newly-singleton cells.  
We need now to write a method `full_propagation` that at the beginning starts with a set of `to_propagate` cells (if it is not specified, then we just take it to consist of all singleton cells).  Then, it picks a cell from the to_propagate set, and propagates from it, adding any newly singleton cell to to_propagate.  Once there are no more cells to be propagated, the method returns. 
If this sounds similar to graph reachability, it is ... because it is!  It is once again the algorithmic pattern of keeping a list of work to be done, then iteratively picking an element from the list, doing it, possibly updating the list of work to be done with new work that has to be done as a result of what we just did, and continuing in this fashion until there is nothing left to do. 
We will let you write this function.
"""

### Define full propagation

def sudoku_full_propagation(self, to_propagate=None):
    """Iteratively propagates from all singleton cells, and from all
    newly discovered singleton cells, until no more propagation is possible.
    @param to_propagate: sets of cells from where to propagate.  If None, propagates
        from all singleton cells. 
    @return: nothing.
    """
    if to_propagate is None:
        to_propagate = {(i, j) for i in range(9) for j in range(9)}
    # This code is the (A) code; will be referenced later.
    while len(to_propagate) > 0:
        ij = to_propagate.pop()
        to_propagate.update(self.propagate_cell(ij))

Sudoku.full_propagation = sudoku_full_propagation

### Tests for full propagation

sd = Sudoku([
    '53__7____',
    '6__195___',
    '_98____6_',
    '8___6___3',
    '4__8_3__1',
    '7___2___6',
    '_6____28_',
    '___419__5',
    '____8__79'
])
sd.full_propagation()
sd.show(details=True)
sdd = Sudoku.from_string('[[[5], [3], [4], [6], [7], [8], [9], [1], [2]], [[6], [7], [2], [1], [9], [5], [3], [4], [8]], [[1], [9], [8], [3], [4], [2], [5], [6], [7]], [[8], [5], [9], [7], [6], [1], [4], [2], [3]], [[4], [2], [6], [8], [5], [3], [7], [9], [1]], [[7], [1], [3], [9], [2], [4], [8], [5], [6]], [[9], [6], [1], [5], [3], [7], [2], [8], [4]], [[2], [8], [7], [4], [1], [9], [6], [3], [5]], [[3], [4], [5], [2], [8], [6], [1], [7], [9]]]')
assert_equal(sd, sdd)

"""We solved our example problem!  Constraint propagation, iterated, led us to the solution!
"""
## Searching for a solution

sd = Sudoku([
    '53__7____',
    '6___95___',
    '_98____6_',
    '8___6___3',
    '4__8_3__1',
    '7___2___6',
    '_6____28_',
    '___41___5',
    '____8__79'
])
sd.show()
sd.full_propagation()
sd.show(details=True)

"""As we see, there are still undetermined values.  We can peek into the detailed state of the solution:

"""

sd.show(details=True)
# Let's save this Sudoku for later.
sd_partially_solved = Sudoku(sd)

"""What can we do when constraint propagation fails? 

Let us implement search with backtracking.  What we need to do is something like this: 

search():
1. propagate constraints.
1. if solved, hoorrayy!
1. if impossible, raise Unsolvable()
1. if not fully solved, pick a cell with multiple digits possible, and iteratively:
 * Assign one of the possible values to the cell. 
 * Call search() with that value for the cell.
 * If Unsolvable is raised by the search() call, move on to the next value.
 * If all values returned Unsolvable (if we tried them all), then we raise Unsolvable.

So we see that search() is a recursive function.  
From the pseudo-code above, we see that it might be better to pick a cell with few values possible at step 4 above, so as to make our chances of success as good as possible.  For instance, it is much better to choose a cell with set $\{1, 2\}$ than one with set $\{1, 3, 5, 6, 7, 9\}$, as the probability of success is $1/2$ in the first case and $1/6$ in the second case. 
Of course, it may be possible to come up with much better heuristics to guide our search, but this will have to do so far. 

One fine point with the search above is the following.  So far, an object has a self.m matrix, which contains the status of the Sudoku solution. 
We cannot simply pass self.m recursively to search(), because in the course of the search and constraint propagation, self.m will be modified, and there is no easy way to keep track of these modifications. 
Rather, we will write search() as a method, and when we call it, we will:

* First, create a copy of the current object via the Sudoku constructor, so we have a copy we can modify. 
* Second, we assign one of the values to the cell, as above; 
* Third, we will call the search() method of that object. 

"""

def sudoku_done(self):
    """Checks whether an instance of Sudoku is solved."""
    for i in range(9):
        for j in range(9):
            if len(self.m[i][j]) > 1:
                return False
    return True

Sudoku.done = sudoku_done


def sudoku_search(self, new_cell=None):
    """Tries to solve a Sudoku instance."""
    to_propagate = None if new_cell is None else {new_cell}
    self.full_propagation(to_propagate=to_propagate)
    if self.done():
        return self # We are a solution
    # We need to search.  Picks a cell with as few candidates as possible.
    candidates = [(len(self.m[i][j]), i, j)
                   for i in range(9) for j in range(9) if len(self.m[i][j]) > 1]
    _, i, j = min(candidates)
    values = self.m[i][j]
    # values contains the list of values we need to try for cell i, j.
    # print("Searching values", values, "for cell", i, j)
    for x in values:
        # print("Trying value", x)
        sd = Sudoku(self)
        sd.m[i][j] = {x}
        try:
            # If we find a solution, we return it.
            return sd.search(new_cell=(i, j))
        except Unsolvable:
            # Go to next value.
            pass
    # All values have been tried, apparently with no success.
    raise Unsolvable()

Sudoku.search = sudoku_search


def sudoku_solve(self, do_print=True):
    """Wrapper function, calls self and shows the solution if any."""
    try:
        r = self.search()
        if do_print:
            print("We found a solution:")
            r.show()
            return r
    except Unsolvable:
        if do_print:
            print("The problem has no solutions")

Sudoku.solve = sudoku_solve

"""Let us try this on our previous Sudoku problem that was not solvable via constraint propagation alone."""

sd = Sudoku([
    '53__7____',
    '6___95___',
    '_98____6_',
    '8___6___3',
    '4__8_3__1',
    '7___2___6',
    '_6____28_',
    '___41___5',
    '____8__79'
])
sd.solve()

"""It works, search with constraint propagation solved the Sudoku puzzle!

## The choice - constraint propagation - recursion paradigm.

We have learned a general strategy for solving difficult problems.  The strategy can be summarized thus: **choice - constraint propagation - recursion.** 

In the _choice_ step, we make one guess from a set of possible guesses.  If we want our search for a solution to be exhaustive, as in the above Sudoku example, we ensure that we try iteratively all choices from a set of choices chosen so that at least one of them must succeed.  In the above example, we know that at least one of the digit values must be the true one, hence our search is exhaustive.  In other cases, we can trade off exhaustiveness for efficiency, and we may try only a few choices, guided perhaps by an heuristic. 

The _constraint propagation_ step propagates the consequences of the choice to the problem.  Each choice thus gives rise to a new problem, which is a little bit simpler than the original one as some of the possible choices, that is, some of its complexity, has been removed.  In the Sudoku case, the new problem has less indetermination, as at least one more of its cells has a known digit in it. 

The problems resulting from _constraint propagation_, while simpler, may not be solved yet.  Hence, we _recur_, calling the solution procedure on them as well.  As these problems are simpler (they contain fewer choices), eventually the recursion must reach a point where no more choice is possible, and whether constraint propagation should yield a completely defined problem, one of which it is possible to say whether it is solvable or not with a trivial test.  This forms the base case for the recursion. 

This solution strategy applies very generally, to problems well beyond Sudoku.

## Part 2: Digits must go somewhere

If you have played Sudoku before, you might have found the way we solved Sudoku puzzles a bit odd. 
The constraint we encoded is: 

> If a digit appears in a cell, it cannot appear anywhere else on the same row, column, or 3x3 block as the cell. 

This _is_ a rule of Sudoku.  Normally, however, we hear Sudoku described in a different way:

> Every column, row, and 3x3 block should contain all the 1...9 digits exactly once.

There are two questions.  The first is: are the two definitions equivalent? 
Well, no; the first definition does not say what the digits are (e.g., does not rule out 0).  But in our Sudoku representation, we _start_ by saying that every cell can contain only one of 1...9.  If every row (or column, or 3x3 block) cannot contain more than one repetition of each digit, and if there are 9 digits and 9 cells in the row (or column, or block), then clearly every digit must appear exactly once in the row (or column, or block).  So once the set of digits is specified, the two definitions are equivalent. 

The second question is: but still, what happens to the method we usually employ to solve Sudoku? 
I generally don't solve Sudoku puzzles by focusing on one cell at a time, and thinking: is it the case that this call can contain only one digit? 
This is the strategy employed by the solver above.  But it is not the strategy I normally use. 
I generally solve Sudoku puzzles by looking at a block (or row, or column), and thinking: let's consider the digit $k$ ($1 \leq k \leq 9$).  Where can it go in the block?  And if I find that the digit can go in one block cell only, I write it there.  
Does the solver work even without this "where can it go" strategy?  And can we make it follow it? 

The solver works even without the "where can it go" strategy because it exaustively tries all possibilities.  This means the solver works without the strategy; it does not say that the solver works _well_ without the strategy. 

We can certainly implement the _where can it go_ strategy, as part of constraint propagation; it would make our solver more efficient.

## Question 3: A better `full_propagation` method
### Not a real question; just copy some previous code into a new method.
There is a subtle point in applying the _where can it go_ heuristics. 

Before, when our only constraint was the uniqueness in each row, column, and block, we needed to propagate only from cells that hold a singleton value. 
If a cell held a non-singleton set of digits, such as $\{2, 5\}$, no values could be ruled out as a consequence of this on the same row, column, or block. 

The _where can it go_ heuristic, instead, benefits from knowing that in a cell, the set of values went for instance from $\{2, 3, 5\}$ to $\{2, 5\}$: by ruling out the possibility of a $3$ in this cell, it may be possibe to deduct that the digit $3$ can appear in only one (other) place in the block, and place it there. 

Thus, we modify the `full_propagation` method.  The method does:
* Repeat:
  * first does propagation as before, based on singletons; 
  * then, it applies the _where can it go_ heuristic on the whole Sudoku board. 
* until there is nothing more that can be propagated. 

Thus, we replace the `full_propagation` method previously defined with this new one, where the (A) block of code is what you previously wrote in `full_propagation`.
You don't need to write new code here: just copy your solution for `full_propagation` into the (A) block below.
"""

### Define full propagation with where can it go

def sudoku_full_propagation_with_where_can_it_go(self, to_propagate=None):
    """Iteratively propagates from all singleton cells, and from all
    newly discovered singleton cells, until no more propagation is possible."""
    if to_propagate is None:
        to_propagate = {(i, j) for i in range(9) for j in range(9)}
    while len(to_propagate) > 0:
        while len(to_propagate) > 0:
            ij = to_propagate.pop()
            to_propagate.update(self.propagate_cell(ij))
        # Now we check whether there is any other propagation that we can
        # get from the where can it go rule.
        to_propagate = self.where_can_it_go()

"""## Question 4: Implement the `occurs_once_in_sets` helper

To implement the `where_can_it_go` method, let us write a helper function, or better, let's have you write it.  Given a sequence of sets $S_1, S_2, \ldots, S_n$, we want to obtain the set of elements that appear in _exactly one_ of the sets (that is, they appear in one set, and _only_ in one set).   Mathematically, we can write this as 
$$
(S_1 \setminus (S_2 \cup \cdots \cup S_n)) \cup (S_2 \setminus (S_1 \cup S_3 \cup \cdots \cup S_n)) \cup \cdots \cup
(S_n \setminus (S_1 \cup \cdots \cup S_{n-1}))
$$
even though that's certainly not the easiest way to compute it!
The problem can be solved with the help of [defaultdict](https://docs.python.org/3/library/collections.html#collections.defaultdict!) to count the occurrences, and is 5 lines long.
Of course, other solutions are possible as well.
"""

### Define helper function to check once-only occurrence

from collections import defaultdict

def occurs_once_in_sets(set_sequence):
    """Returns the elements that occur only once in the sequence of sets set_sequence.
    The elements are returned as a set."""
    # YOUR CODE HERE
    final = set()
    discarded = set()
    for i in set_sequence:
        for j in i:
            if j not in discarded:
                if j in final:
                    final.discard(j)
                    discarded.add(j)
                else:
                    final.add(j)
    return final


### Tests for once-only

from nose.tools import assert_equal

assert_equal(occurs_once_in_sets([{1, 2}, {2, 3}]), {1, 3})
assert_equal(occurs_once_in_sets([]), set())
assert_equal(occurs_once_in_sets([{2, 3, 4}]), {2, 3, 4})
assert_equal(occurs_once_in_sets([set()]), set())
assert_equal(occurs_once_in_sets([{2, 3, 4, 5, 6}, {5, 6, 7, 8}, {5, 6, 7}, {4, 6, 7}]), {2, 3, 8})

"""## Question 5: Implement _where can it go_. 

We are now ready to write -- or better, to have you write -- the _where can it go_ method.  
The method is global: it examines all rows, all columns, and all blocks.  
If it finds that in a row (or column, or block), a value can fit in only one cell, and that cell is not currently a singleton (for otherwise there is nothing to be done), it sets the value in the cell, and it adds the cell to the newly_singleton set that is returned, just as in propagate_cell. 
The portion of method that you need to write is about two dozen lines of code long.
"""

### Write where_can_it_go

def sudoku_where_can_it_go(self):
    """Sets some cell values according to the where can it go
    heuristics, by examining all rows, colums, and blocks."""
    newly_singleton = set()


    for row in range(9):
        x = occurs_once_in_sets(self.m[row])
        for i in x:
            for j in range(9):
                if i in self.m[row][j] and len(self.m[row][j]) != 1:
                    self.m[row][j] = {i}
                    newly_singleton.add((row,j))

        l = []
        for col in range(9):
            l.append(self.m[col][row])
        
        x = occurs_once_in_sets(l)
        for i in x:
            for j in range(9):
                if i in self.m[j][row] and len(self.m[j][row]) != 1:
                    self.m[j][row] = {i}
                    newly_singleton.add((j, row))
    

    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            l2 = []
            for ii in range(3):
                for jj in range(3):
                    l2.append(self.m[ii + 3 * (i // 3)][jj + 3 * (j // 3)])                    
            x = occurs_once_in_sets(l2)
            for z in x:
                for ii in range(3):
                    for jj in range(3):
                        r = ii + 3 * (i // 3)
                        c = jj + 3 * (j // 3)
                        if z in self.m[r][c] and len(self.m[r][c]) != 1:
                            self.m[r][c] = {z}
                            newly_singleton.add((r, c))


    # Returns the list of newly-singleton cells.
    return newly_singleton

Sudoku.where_can_it_go = sudoku_where_can_it_go


### Tests for where can it go

sd = Sudoku.from_string('[[[5], [3], [1, 2, 4], [1, 2, 6], [7], [1, 2, 6, 8], [4, 8, 9], [1, 2, 4], [2, 8]], [[6], [1, 4, 7], [1, 2, 4, 7], [1, 2, 3], [9], [5], [3, 4, 8], [1, 2, 4], [2, 7, 8]], [[1, 2], [9], [8], [1, 2, 3], [4], [1, 2], [3, 5], [6], [2, 7]], [[8], [1, 5], [1, 5, 9], [1, 7, 9], [6], [1, 4, 7, 9], [4, 5], [2, 4, 5], [3]], [[4], [2], [6], [8], [5], [3], [7], [9], [1]], [[7], [1, 5], [1, 3, 5, 9], [1, 9], [2], [1, 4, 9], [4, 5, 8], [4, 5], [6]], [[1, 9], [6], [1, 5, 7, 9], [5, 7, 9], [3], [7, 9], [2], [8], [4]], [[2, 9], [7, 8], [2, 7, 9], [4], [1], [2, 7, 9], [6], [3], [5]], [[2, 3], [4, 5], [2, 3, 4, 5], [2, 5, 6], [8], [2, 6], [1], [7], [9]]]')
print("Original:")
sd.show(details=True)
new_singletons = set()

while True:
    new_s = sd.where_can_it_go()
    if len(new_s) == 0:
        break
    new_singletons |= new_s
assert_equal(new_singletons,
             {(3, 2), (2, 6), (7, 1), (5, 6), (2, 8), (8, 0), (0, 5), (1, 6),
              (2, 3), (3, 7), (0, 3), (5, 1), (0, 8), (8, 5), (5, 3), (5, 5),
              (8, 1), (5, 7), (3, 1), (0, 6), (1, 8), (3, 6), (5, 2), (1, 1)})
print("After where can it go:")
sd.show(details=True)
sdd = Sudoku.from_string('[[[5], [3], [1, 2, 4], [6], [7], [8], [9], [1, 2, 4], [2]], [[6], [7], [1, 2, 4, 7], [1, 2, 3], [9], [5], [3], [1, 2, 4], [8]], [[1, 2], [9], [8], [3], [4], [1, 2], [5], [6], [7]], [[8], [5], [9], [1, 9, 7], [6], [1, 4, 9, 7], [4], [2], [3]], [[4], [2], [6], [8], [5], [3], [7], [9], [1]], [[7], [1], [3], [9], [2], [4], [8], [5], [6]], [[1, 9], [6], [1, 5, 9, 7], [9, 5, 7], [3], [9, 7], [2], [8], [4]], [[9, 2], [8], [9, 2, 7], [4], [1], [9, 2, 7], [6], [3], [5]], [[3], [4], [2, 3, 4, 5], [2, 5, 6], [8], [6], [1], [7], [9]]]')
print("The above should be equal to:")
sdd.show(details=True)
assert_equal(sd, sdd)

sd = Sudoku([
    '___26_7_1',
    '68__7____',
    '1____45__',
    '82_1___4_',
    '__46_2___',
    '_5___3_28',
    '___3___74',
    '_4__5__36',
    '7_3_18___'
])
print("Another Original:")
sd.show(details=True)
print("Propagate once:")
sd.propagate_all_cells_once()
# sd.show(details=True)
new_singletons = set()
while True:
    new_s = sd.where_can_it_go()
    if len(new_s) == 0:
        break
    new_singletons |= new_s
print("After where can it go:")
sd.show(details=True)
sdd = Sudoku.from_string('[[[4], [3], [5], [2], [6], [9], [7], [8], [1]], [[6], [8], [2], [5], [7], [1], [4], [9], [3]], [[1], [9], [7], [8], [3], [4], [5], [6], [2]], [[8], [2], [6], [1], [9], [5], [3], [4], [7]], [[3], [7], [4], [6], [8], [2], [9], [1], [5]], [[9], [5], [1], [7], [4], [3], [6], [2], [8]], [[5], [1], [1, 2, 5, 6, 8, 9], [3], [2], [6], [1, 2, 8, 9], [7], [4]], [[2], [4], [1, 2, 8, 9], [9], [5], [7], [1, 2, 8, 9], [3], [6]], [[7], [6], [3], [4], [1], [8], [2], [5], [9]]]')
print("The above should be equal to:")
sdd.show(details=True)
assert_equal(sd, sdd)

"""Let us try it now on a real probem. Note from before that this Sudoku instance could not be solved via propagate_cells alone:"""

sd = Sudoku(sd_partially_solved)
newly_singleton = sd.where_can_it_go()
print("Newly singleton:", newly_singleton)
print("Resulting Sudoku:")
sd.show(details=True)

"""As we can see, the heuristics led to substantial progress.   Let us incorporate it in the Sudoku solver. """

Sudoku.full_propagation = sudoku_full_propagation_with_where_can_it_go

"""Let us try again to solve a Sudoku example which, as we saw before, could not be solved by constrain propagation only (without using the _where can it go_ heuristics).  Can we solve it now via constraint propagation?"""

sd = Sudoku([
    '53__7____',
    '6___95___',
    '_98____6_',
    '8___6___3',
    '4__8_3__1',
    '7___2___6',
    '_6____28_',
    '___41___5',
    '____8__79'
])
print("Initial:")
sd.show()
sd.full_propagation()
print("After full propagation with where can it go:")
sd.show()

"""No!  We still cannot! But if we compare the above with the previous attempt, we see that the heuristic led to much more progress; very few positions still remain to be determined via search.

## Question 6: Solving some problems from example sites

Let us see how long it takes us to solve examples found around the Web. 
We consider a few from [this site](https://dingo.sbs.arizona.edu/~sandiway/sudoku/examples.html).
You should be able to complete all of these tests in a short amount of time.
"""

import time

"""### Daily Telegraph January 19th "Diabolical"


"""

sd = Sudoku([
    '_2_6_8___',
    '58___97__',
    '____4____',
    '37____5__',
    '6_______4',
    '__8____13',
    '____2____',
    '__98___36',
    '___3_6_9_'
])
t = time.time()
sd.solve()
elapsed = time.time() - t
print("Solved in", elapsed, "seconds")

assert elapsed < 5

"""### Vegard Hanssen puzzle 2155141"""

sd = Sudoku([
    '___6__4__',
    '7____36__',
    '____91_8_',
    '_________',
    '_5_18___3',
    '___3_6_45',
    '_4_2___6_',
    '9_3______',
    '_2____1__'
])
t = time.time()
sd.solve()
elapsed = time.time() - t
print("Solved in", elapsed, "seconds")
assert elapsed < 5

"""### A supposedly even harder one

[source](http://www.sudokuwiki.org/Weekly_Sudoku.asp?puz=28)
"""

sd = Sudoku([
    '6____894_',
    '9____61__',
    '_7__4____',
    '2__61____',
    '______2__',
    '_89__2___',
    '____6___5',
    '_______3_',
    '8____16__'
])
t = time.time()
sd.solve()
elapsed = time.time() - t
print("Solved in", elapsed, "seconds")
assert elapsed < 10

"""## Trying puzzles in bulk

Let us try the puzzles found at [https://raw.githubusercontent.com/shadaj/sudoku/master/sudoku17.txt](https://raw.githubusercontent.com/shadaj/sudoku/master/sudoku17.txt); apparently lines 517 and 6361 are very hard). 
"""

import requests

r = requests.get("https://raw.githubusercontent.com/shadaj/sudoku/master/sudoku17.txt")
puzzles = r.text.split()

"""Let us convert these puzzles to our format."""

def convert_to_our_format(s):
    t = s.replace('0', '_')
    r = []
    for i in range(9):
        r.append(t[i * 9: (i + 1) * 9])
    return r

"""You need to solve these tests efficiently."""


t = 0
max_d = 0.
max_i = None
t = time.time()
for i, s in enumerate(puzzles[:1000]):
    p = convert_to_our_format(puzzles[i])
    sd = Sudoku(p)
    sd.solve(do_print=False)
elapsed = time.time() - t
print("It took you", elapsed, "to solve the first 1000 Sudokus.")
assert elapsed < 30