for func in branin ackley5
do
    for method in EI 2.wsms.gh.1 3.wsms.gh.1
    do
        python run_bo_task.py $func $method 2 2 1 debug_results log 
    done
done
