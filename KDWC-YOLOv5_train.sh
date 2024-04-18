#!/bin/bash
#SBATCH -J czh
#SBATCH -p p-A100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:1
#SBATCH --nodelist=pgpu16
#SBATCH -o out.%j
#SBATCH -e err.%j
#SBATCH -A P00120210009

##################################################################

module load cuda11.3/toolkit/11.3.0
function Func1(){
    cal=1
    sleep 5
    while true
    do 
    nvidia-smi
    cal=$(($cal+1))
    if [ $cal -gt 10 ]
    then break
    fi
    sleep 2
    done
}

function Func2(){
    cd detection/yolov5/
    python train.py --multi-scale
}

Func1&Func2

