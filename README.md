Planet_Lab_v33_withJ2_Neptune.py requires the Planet_Lab_v33_fnc, the calculate_J2.py file, and the hydrogen equation of state table.

Run the program from the command line using python3 or similar.  A GUI appears with preset values that can be altered in the various
input cells.  Start the calculation by clicking [START].  The program iterates to find the solution for a given H2 mass fraction and takes
a number of minutes to run depending on the system resoures.  Output is sent to a directory named for the input mass and surface pressure.

Output includes a series of .txt files containing the model results and a series of plots, all sent to the output directory.  

Two versions of Planet_Lab are included, one with sub-Neptune-like inputs preloaded, and the other setup for Neptune. The Neptune version
outputs model gravitational parameters. 
