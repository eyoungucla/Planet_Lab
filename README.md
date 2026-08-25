Planet_Lab_v33_f.py requires the core function Planet_Lab_v33_fnc_f.py, the fortran module for the binodal compiled from binodal_f.f90, and the hydrogen equation of state table. The inclusion of the fortran binodal greatly improves the calculation speed for an individual model. 

Run the program from the command line using python3 or similar.  A GUI appears with preset values that can be altered in the various
input cells.  Start the calculation by clicking [START].  The program iterates to find the solution for a given H2 mass fraction and takes
a number of minutes to run depending on the system resoures.  Output is sent to a directory named for the input mass and surface pressure.

Output includes a series of .txt files containing the model results and a series of plots, all sent to the output directory.  

