def calc_func(x, y, comm_world, time_array, **kwargs):
    """
    This is what BayesianOptimization calls.
    It takes parameters (x, y from Bayes), 
    does the calculation, 
    returns ONE score.
    """
    playdo = compute_playdo(x, y, **kwargs)
    
    # Send b_line work to other nodes via MPI
    result = b_line(
        playdo=playdo, 
        param1=kwargs.get('param1'),
        param2=kwargs.get('param2'),
        param3=kwargs.get('param3'),
        comm_world=comm_world,
        time_array=time_array  # This is probably what gets distributed
    )
    
    # Extract ONE metric from result to return
    score = extract_metric(result)  # e.g., mean, max, some KPI
    return score


def b_line(playdo, param1, param2, param3, comm_world, time_array, **kwargs):
    rank = comm_world.Get_rank()
    size = comm_world.Get_size()
    
    # Split time_array across ranks
    local_time_array = time_array[rank::size]  # Each rank gets every Nth element
    
    # Do computation on local chunk
    local_result = compute_on_chunk(playdo, param1, param2, param3, local_time_array)
    
    # Gather results back to rank 0
    all_results = comm_world.gather(local_result, root=0)
    
    if rank == 0:
        final_result = aggregate(all_results)
        return final_result  # This goes back to calc_func as the score


def main():
    # Load YAML config
    config = load_yaml('file_config.yaml')
    
    pbounds = {
        'x': (config['x_min'], config['x_max']),
        'y': (config['y_min'], config['y_max']),
    }
    
    optimizer = BayesianOptimization(
        f=lambda x, y: calc_func(
            x=x, 
            y=y, 
            comm_world=MPI.COMM_WORLD,
            time_array=config['time_array'],
            param1=config['param1'],
            param2=config['param2'],
            param3=config['param3'],
        ),
        pbounds=pbounds,
        random_state=1,
    )
    
    # This runs 10 iterations automatically
    # Each iteration: pick (x,y) → calc_func → score → use score to pick next (x,y)
    optimizer.maximize(n_iter=10, init_points=5)
    
    # Get best result
    best = optimizer.max
    print(f"Best params: {best['params']}")
    print(f"Best score: {best['target']}")
