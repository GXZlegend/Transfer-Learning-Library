CUDA_VISIBLE_DEVICES=1 python shot.py /data/office-home/ --domain Cl --wn --val-ratio 0.1 --epochs 100 --log ./shot_Cl_0/ --phase train_source --lb-smooth 0.1
CUDA_VISIBLE_DEVICES=1 python shot.py /data/office-home/ --domain Ar --wn --val-ratio 0   --epochs 15  --log ./shot_Cl_Ar_0/ --phase train_target --load ./shot_Cl_0/checkpoints/best.pth --trade-off 0.3
CUDA_VISIBLE_DEVICES=1 python shot.py /data/office-home/ --domain Pr --wn --val-ratio 0   --epochs 15  --log ./shot_Cl_Pr_0/ --phase train_target --load ./shot_Cl_0/checkpoints/best.pth --trade-off 0.3
CUDA_VISIBLE_DEVICES=1 python shot.py /data/office-home/ --domain Rw --wn --val-ratio 0   --epochs 15  --log ./shot_Cl_Rw_0/ --phase train_target --load ./shot_Cl_0/checkpoints/best.pth --trade-off 0.3
CUDA_VISIBLE_DEVICES=1 python shot.py /data/office-home/ --domain Ar --wn --val-ratio 0   --epochs 15  --log ./shotim_Cl_Ar_0/ --phase train_target --load ./shot_Cl_0/checkpoints/best.pth --trade-off 0
CUDA_VISIBLE_DEVICES=1 python shot.py /data/office-home/ --domain Pr --wn --val-ratio 0   --epochs 15  --log ./shotim_Cl_Pr_0/ --phase train_target --load ./shot_Cl_0/checkpoints/best.pth --trade-off 0
CUDA_VISIBLE_DEVICES=1 python shot.py /data/office-home/ --domain Rw --wn --val-ratio 0   --epochs 15  --log ./shotim_Cl_Rw_0/ --phase train_target --load ./shot_Cl_0/checkpoints/best.pth --trade-off 0

CUDA_VISIBLE_DEVICES=1 python shot.py /data/office-home/ --domain Cl --wn --val-ratio 0.1 --epochs 100 --log ./shot_Cl_1/ --phase train_source --lb-smooth 0.1
CUDA_VISIBLE_DEVICES=1 python shot.py /data/office-home/ --domain Ar --wn --val-ratio 0   --epochs 15  --log ./shot_Cl_Ar_1/ --phase train_target --load ./shot_Cl_1/checkpoints/best.pth --trade-off 0.3
CUDA_VISIBLE_DEVICES=1 python shot.py /data/office-home/ --domain Pr --wn --val-ratio 0   --epochs 15  --log ./shot_Cl_Pr_1/ --phase train_target --load ./shot_Cl_1/checkpoints/best.pth --trade-off 0.3
CUDA_VISIBLE_DEVICES=1 python shot.py /data/office-home/ --domain Rw --wn --val-ratio 0   --epochs 15  --log ./shot_Cl_Rw_1/ --phase train_target --load ./shot_Cl_1/checkpoints/best.pth --trade-off 0.3
CUDA_VISIBLE_DEVICES=1 python shot.py /data/office-home/ --domain Ar --wn --val-ratio 0   --epochs 15  --log ./shotim_Cl_Ar_1/ --phase train_target --load ./shot_Cl_1/checkpoints/best.pth --trade-off 0
CUDA_VISIBLE_DEVICES=1 python shot.py /data/office-home/ --domain Pr --wn --val-ratio 0   --epochs 15  --log ./shotim_Cl_Pr_1/ --phase train_target --load ./shot_Cl_1/checkpoints/best.pth --trade-off 0
CUDA_VISIBLE_DEVICES=1 python shot.py /data/office-home/ --domain Rw --wn --val-ratio 0   --epochs 15  --log ./shotim_Cl_Rw_1/ --phase train_target --load ./shot_Cl_1/checkpoints/best.pth --trade-off 0

CUDA_VISIBLE_DEVICES=1 python shot.py /data/office-home/ --domain Cl --wn --val-ratio 0.1 --epochs 100 --log ./shot_Cl_2/ --phase train_source --lb-smooth 0.1
CUDA_VISIBLE_DEVICES=1 python shot.py /data/office-home/ --domain Ar --wn --val-ratio 0   --epochs 15  --log ./shot_Cl_Ar_2/ --phase train_target --load ./shot_Cl_2/checkpoints/best.pth --trade-off 0.3
CUDA_VISIBLE_DEVICES=1 python shot.py /data/office-home/ --domain Pr --wn --val-ratio 0   --epochs 15  --log ./shot_Cl_Pr_2/ --phase train_target --load ./shot_Cl_2/checkpoints/best.pth --trade-off 0.3
CUDA_VISIBLE_DEVICES=1 python shot.py /data/office-home/ --domain Rw --wn --val-ratio 0   --epochs 15  --log ./shot_Cl_Rw_2/ --phase train_target --load ./shot_Cl_2/checkpoints/best.pth --trade-off 0.3
CUDA_VISIBLE_DEVICES=1 python shot.py /data/office-home/ --domain Ar --wn --val-ratio 0   --epochs 15  --log ./shotim_Cl_Ar_2/ --phase train_target --load ./shot_Cl_2/checkpoints/best.pth --trade-off 0
CUDA_VISIBLE_DEVICES=1 python shot.py /data/office-home/ --domain Pr --wn --val-ratio 0   --epochs 15  --log ./shotim_Cl_Pr_2/ --phase train_target --load ./shot_Cl_2/checkpoints/best.pth --trade-off 0
CUDA_VISIBLE_DEVICES=1 python shot.py /data/office-home/ --domain Rw --wn --val-ratio 0   --epochs 15  --log ./shotim_Cl_Rw_2/ --phase train_target --load ./shot_Cl_2/checkpoints/best.pth --trade-off 0

