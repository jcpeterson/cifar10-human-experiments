#!/usr/bin/env bash
echo 'outer tuning script'

models='resnet_preact_bottleneck_164' # no commas!
#declare -a models=(shake_shake_26_2x64d_SSI_cutout16)

declare -A archs=([vgg_15_BN_64]=vgg [resnet_basic_110]=resnet [resnet_preact_bottleneck_164]=resnet_preact [wrn_28_10]=wrn [densenet_BC_100_12]=densenet [pyramidnet_basic_110_270]=pyramidnet [resnext_29_8x64d]=resnext [wrn_28_10_cutout16]=wrn [shake_shake_26_2x64d_SSI_cutout16]=shake_shake)

declare -a control=(True False)

declare -a lr=(0.1 0.01 0.001)

declare -a seeds=(0 1 2)
# for every model
#for model in "${!archs[@]}"
for model in $models
    do
    echo $model
    echo ${archs[$model]}
    arch=${archs[$model]}
    echo $arch
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
                identifier=con_${con}_lr_${l}_seed_${s} 
                logfile=${model}_control:${con}_lr:${l}_seed:${s}.out
                interval=2
#                echo $s
                python_args="--arch=${arch} --c10h_save_interval=${interval} --dataset=CIFAR10H --no_output --c10h_datasplit_seed=${s} --human_tune --nonhuman_control=${con} --base_lr=${l}"
                echo 'python args: '"${python_args}"
                sbatch --output=${logfile} --export=model=$model,identifier=${identifier},python_args="${python_args}",logfile=${logfile} tune_bash_inner.sh
                done
            done


        done
    done

echo 'outer done'
