### Code Analysis Results
#### Overall Findings

* The project is structured with clear
separation of concerns.
* There is a good use of functions to
encapsulate logic.
* The code is mostly readable and
maintainable.

#### Dependencies Used
* The following libraries have been
identified as dependencies:
* glob (*)
* greputil (*)

#### Code Smells Identified

### Code Smells to be Addressed
- Too many global variables (5)
- Redundant variable declarations (15)
- Inconsistent naming conventions used
throughout the project.

### Recommendations
1.  Reduce the number of global variables
by encapsulating them in a separate module
and importing where necessary.
2.  Use consistent naming conventions for
modules, functions, and variables across the
entire project.
3.  Remove redundant variable declarations
to declutter the code.



Note: `(*.) indicates files that require
specific modifications`
### Implementation Steps
* For removing global variables:
+ Create a new file and define them as properties of an object or class
+ Modify all references to update accordingly
* For adopting consistent naming conventions:
+ Choose one convention throughout.
+ Update code to fit the selected convention.
