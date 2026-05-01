BayesianOptimization.maximize() calls your black_box_function
    ↓
black_box_function = calc_func() 
    ↓
calc_func creates playdo, calls b_line(playdo, params, comm_world, ...)
    ↓
b_line() does distributed calculations on other nodes via MPI
    ↓
b_line() returns a SINGLE SCALAR VALUE (the "score")
    ↓
That score goes BACK to BayesianOptimization
    ↓
BayesianOptimization uses that score to decide what params to try NEXT

So what I have is 
I've currently got a workflow that I want to implement with MPI somehow
I use a .sh file that runs a container on a slurm cluster - all nodes can read from the head node.  I need this also to start up with either srun or mpiexec or something~
The .sh file does something like apptainer exec /path/to/container python some_file_config.yaml
some has various functions
first, main()
this takes in some yaml params from the file_config.yaml
after that, we have a Bayes optimization function that takes a several yaml params from that file_config and one of the inputs is a function, called calc_func() in the top portion of the file, using parallel and some more params~.
----------------
Outside of main, where all the functions are made - we basically have calc_func().  This takes in params. 
After taking in params, it gets some value, playdo, that is then fed into another function.  This function is called b_line().
b_line(playdo=playdo, param1, param2, param3, comm_world, **kwargs)
I need this function to be sent off to other nodes/cores to do calculations. For this, do I need a specific variable that is passed in to be "divvy'd up" for the other nodes? I believe there is a time_array but I'm not sure what the contents are. 
After the calculation is done, it currently puts the output into a csv with a uuid.  Then I get a large list of csv files.
Eventually I'll concat them all with the head node.
How I believe this needs to work is - the bayes optimization function should run multiple times, say 10 (using like BayesianOpimization.maximize(n_iter=10, init_points=5).  Each time, it gets a new result and that's passed in as parameters into the calc_func where those parameters are taken in to create the playdo variable.  And that's passed into the b_line function.  Then the csvs are created, sent to head node or master and then it populates some global csv. 
I guess I'm confused, somehow values in b_line function are supposed to inform the next iteration but idk how it's supposed to takes some new value from there and puts it through the bayes optimization function for the 2nd time.
