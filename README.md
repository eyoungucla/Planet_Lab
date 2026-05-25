Planet_Lab_v33_withJ2.py requires the core function Planet_Lab_v33_fnc.py, and ancillary functions calculate_J2.py file, the various cms functions, and the hydrogen equation of state table. The accuracy of the cms run at the end of the model depends on settings for grid resolution.  

Run the program from the command line using python3 or similar.  A GUI appears with preset values that can be altered in the various
input cells.  Start the calculation by clicking [START].  The program iterates to find the solution for a given H2 mass fraction and takes
a number of minutes to run depending on the system resoures.  Output is sent to a directory named for the input mass and surface pressure.

Output includes a series of .txt files containing the model results and a series of plots, all sent to the output directory.  

