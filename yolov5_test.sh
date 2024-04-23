#!/bin/bash
#SBATCH -J KDWC-Yolov5
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:1
#SBATCH -o out.%j
#SBATCH -e err.%j
#SBATCH -A XXXXX
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
    python val.py --weights /mntcephfs/lab_data/wangcm/czh/detection/yolov5/runs/train/exp2/weights/best.pt --data /mntcephfs/lab_data/wangcm/czh/detection/yolov5/data/A30.yaml --task test
}

Func1&Func2

