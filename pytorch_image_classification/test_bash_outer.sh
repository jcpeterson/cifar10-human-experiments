#!/usr/bin/env bash
echo 'outer tuning script'

models='vgg_15_BN_64'
#declare -a models=(shake_shake_26_2x64d_SSI_cutout16)

declare -A archs=([vgg_15_BN_64]=vgg [resnet_basic_110]=resnet [resnet_preact_bottleneck_164]=resnet [wrn_28_10]=wrn [densenet_BC_100_12]=dense [pyramidnet_basic_110_270]=pyramidnet [resnext_29_8x64d]=resnext [wrn_28_10_cutout16]=wrn [shake_shake_26_2x64d_SSI_cutout16]=shake)

# for every model
#for model in "${!archs[@]}"
for model in $models
    echo $model
    arch=${archs[$model]}
    logfile=${model}_control:${con}_lr:${l}_seed:${s}.out
                 interval=2
#                echo $s
                python_args="--arch=${arch} --c10h_save_interval=${interval} --dataset=CIFAR10H --no_output --c10h_datasplit_seed=${s} --human_tune --nonhuman_control=${con} --base_lr=${l}"
                echo 'python args: '"${python_args}"
                sbatch --output=${logfile} --export=model=$model,python_args="${python_args}",logfile=${logfile} tune_bash_inner.sh
                done
            done


        done
    done

echo 'outer done'
