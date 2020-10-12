#for func in branin svm_on_grid
#do
#    for method in EI 2.EI.best 2.EI.sample 2.ts.10 3.tss.10 2.ms.10 2.qEI 2.wsms.10 #2.glasses 2.rollout.2 3.rollout.2
#    do
#        python run_bo_task.py $func $method 2 2 1 debug_results debug_log
#    done
#done

for func in branin ackley5
do
    for method in 2.wsms.gh.10 3.wsms.gh.10.5 2.wstbps.gh.10 3.wstbps.gh.10.5
    do
        nohup python run_bo_task.py $func $method 2 2 1 debug_results log &
    done
done