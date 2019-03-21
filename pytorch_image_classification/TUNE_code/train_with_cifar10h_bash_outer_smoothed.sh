#!/usr/bin/env bash
echo 'outer tuning script'

models='vgg_15_BN_64
shake_shake_26_2x64d_SSI_cutout16
resnet_basic_110
resnet_preact_bottleneck_164
densenet_BC_100_12
wrn_28_10
pyramidnet_basic_110_270
resnext_29_8x64d
wrn_28_10_cutout16' # no commas!
# maybe no quotes needed below?

declare -A params=([vgg_15_BN_64]="--arch=vgg"
[resnet_basic_110]="--arch=resnet --depth 110 --block_type basic"
[resnet_preact_bottleneck_164]="--arch=resnet_preact --depth 164 --block_type bottleneck"
[wrn_28_10]="--arch=wrn --depth 28 --widening_factor 10"
[densenet_BC_100_12]="--arch=densenet --depth 100 --block_type=bottleneck --growth_rate=12 --compression_rate=0.5"
[pyramidnet_basic_110_270]="--arch=pyramidnet --depth 110 --block_type=basic --pyramid_alpha=270"
[resnext_29_8x64d]="--arch=resnext --depth 29 --cardinality=8 --base_channels=64 --batch_size=64 --base_lr=0.05"
[wrn_28_10_cutout16]="--arch=wrn --depth 28 --epochs=200 --scheduler=cosine --base_lr=0.1 --batch_size=64 --seed 17 --use_cutout --cutout_size 16"
[shake_shake_26_2x64d_SSI_cutout16]="--arch=shake_shake --depth=26 --base_channels=64 --epochs=300 --scheduler=cosine --base_lr=0.1 --seed 17 --batch_size=64 --use_cutout --cutout_size=16")

declare -A times=([vgg_15_BN_64]=200
[resnet_basic_110]=200
[resnet_preact_bottleneck_164]=200
[wrn_28_10]=200
[densenet_BC_100_12]=200
[pyramidnet_basic_110_270]=200
[resnext_29_8x64d]=200
[wrn_28_10_cutout16]=200
[shake_shake_26_2x64d_SSI_cutout16]=200)
echo 'declared dicts'

declare -a control=(True False)
#declare -a control=(True)

#declare -a lr=(0.1 0.01 0.001 0.0001 0.00001)
#declare -a lr=(0.1 0.01) # first run
declare -a lr=(0.1)

#declare -a seeds=(0 1 2)
declare -a seeds=(0)

declare -a folds=(0 1 2 3 4 5 6 7 8 9)
# declare -a folds=(0 1 2 3 4)
#declare -a folds=(0)

# for every model
for model in $models
    do
    time=${times[$model]}

    echo $model
    pars=${params[$model]}
    echo $params
    
    # for control or not
    for con in "${control[@]}"
        do
        # for every lr
        for l in "${lr[@]}"
            do
            # use different seeds
            for s in "${seeds[@]}"
                do
                for fold in "${folds[@]}"
                    do
                    identifier=con_${con}_lr_${l}_seed_${s}_fold_${fold}
                    echo 'identifier'
                    echo ${identifier} 
                    logfile=${model}_control:${con}_lr:${l}_seed:${s}_fold:${fold}.out
                    interval=1
                    python_args="${pars} --c10h_save_interval=${interval} --dataset=CIFAR10H --no_output --c10h_datasplit_seed=${s} --human_tune --nonhuman_control=${con} --base_lr=${l} --cv_index=${fold}"
                    echo 'python args: '"${python_args}"
                    sbatch --time=${time}  --output=${logfile} --export=model=$model,identifier=${identifier},python_args="${python_args}",logfile=${logfile} train_with_cifar10h_bash_inner_smoothed.sh
                    done
                done
            done


        done
    done

echo 'outer done'
