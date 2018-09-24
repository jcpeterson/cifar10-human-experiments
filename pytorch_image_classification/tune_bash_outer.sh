#!/usr/bin/env bash
echo 'outer tuning script'

#declare -a models=('vgg_15_BN_64' 'resnet_basic_110' 
#'resnet_preact_bottleneck_164' 'wrn_28_10 densenet_BC_100_12' 
#'pyramidnet_basic_110_270' 'resnext_29_8x64d' 
#'wrn_28_10_cutout16' 'shake_shake_26_2x64d_SSI_cutout16')

declare -a models=('vgg_15_BN_64')

declare -a control=(True False)

declare -a lr=(0.1 0.01 0.001)

declare -a seeds=(0 1 2)
# for every model
for model in "${models[@]}"
    do
#    echo $model
    # for control or not
    for con in "${control[@]}"
        do
#        echo $con
       
        # for every lr
        for l in "${lr[@]}"
            do
#            echo $l

            # use different seeds
            for s in "${seeds[@]}"
                do
                 logfile=${model}_control:${con}_lr:${l}_seed:${s}.out

#                echo $s
                python_args="--c10h_datasplit_seed=${s} --nonhuman_control=${con} --base_lr=${l}"
                echo 'python args: '"${python_args}"
                sbatch --output=${logfile} --export=model=$model,python_args="${python_args}",logfile=${logfile} tune_bash_inner.sh
                done
            done


        done
    done

echo 'outer done'
