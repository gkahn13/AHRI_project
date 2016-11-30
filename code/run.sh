#for i in {0..3}; do
#    ipython -m main train lr -- -yaml "yamls/lr_exp${i}.yaml"
#done

#for i in {1..1}; do
#    ipython -m main train nn -- -yaml "yamls/nn_exp${i}.yaml"
#done

for i in {4..7}; do
    ipython -m main train bd_nn -- -yaml "yamls/bd_nn_exp${i}.yaml"
done

#for i in {0..3}; do
#    ipython -m main train gp -- -yaml "yamls/gp_exp${i}.yaml"
#done

#for i in {0..3}; do
#    ipython -m main train gp_nn -- -yaml "yamls/gp_nn_exp${i}.yaml"
#done
