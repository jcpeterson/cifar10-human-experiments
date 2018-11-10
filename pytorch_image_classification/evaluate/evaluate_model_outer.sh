#!/usr/bin/env bash
echo 'outer tuning script'

models='pyramidnet_basic_110_270 shake_shake_26_2x64d_SSI_cutout16 wrn_28_10 
resnet_preact_bottleneck_164 resnet_basic_110 vgg_15_BN_64 
densenet_BC_100_12 resnext_29_8x64d wrn_28_10_cutout16'

declare -A archs=([vgg_15_BN_64]=vgg [resnet_basic_110]=resnet [resnet_preact_bottleneck_164]=resnet_preact [wrn_28_10]=wrn [densenet_BC_100_12]=densenet [pyramidnet_basic_110_270]=pyramidnet [resnext_29_8x64d]=resnext [wrn_28_10_cutout16]=wrn [shake_shake_26_2x64d_SSI_cutout16]=shake_shake)

#runs='1 2 3 4' 
runs='2 3 4' 

# for every model
for run in $runs
    do
    for model in $models
        do
        echo $model
        echo ${archs[$model]}
        arch=${archs[$model]}
        echo $arch
        logfile=${model}.pretrained_accuracy.run_${run}.out
    #    python_args="--arch=${arch} --c10h_save_interval=1 --dataset=CIFAR10 --no_output --c10h_datasplit_seed=0 --human_tune --nonhuman_control=${con} --base_lr=${l}"
        python_args="--arch=${arch} --dataset=CIFAR10 --no_output"
        echo 'python args: '"${python_args}"
        sbatch --output=${logfile} --export=run=$run,model=$model,python_args="${python_args},logfile=${logfile}" evaluate_model_inner.sh
        done
    done    
echo 'outer done'
