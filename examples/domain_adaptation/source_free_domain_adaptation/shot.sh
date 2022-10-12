# python shot.py /data/office-home/ --domain Ar --wn --val-ratio 0.1 --train-resizing ran.crop --epochs 50 --log ./shot_Ar_0_1011/ --phase train_source --lb-smooth 0.1 --seed 1011
# python shot.py /data/office-home/ --domain Cl --wn --val-ratio 0   --train-resizing ran.crop --epochs 15 --log ./shot_Ar_Cl_0_1011/ --phase train_target --load ./shot_Ar_0_1011/checkpoints/best.pth --trade-off 0.3 --seed 1011
python shot.py /data/office-home/ --domain Cl --wn --val-ratio 0   --train-resizing ran.crop --epochs 15 --log ./shotim_Ar_Cl_0_1011/ --phase train_target --load ./shot_Ar_0_1011/checkpoints/best.pth --trade-off 0 --seed 1011

# python shot.py /data/office-home/ --domain Ar --wn --val-ratio 0.1 --train-resizing ran.crop --epochs 50 --log ./shot_Ar_1/ --phase train_source --lb-smooth 0.1
# python shot.py /data/office-home/ --domain Cl --wn --val-ratio 0   --train-resizing ran.crop --epochs 15 --log ./shot_Ar_Cl_1/ --phase train_target --load ./shot_Ar_1/checkpoints/best.pth --trade-off 0.3
# python shot.py /data/office-home/ --domain Cl --wn --val-ratio 0   --train-resizing ran.crop --epochs 15 --log ./shotim_Ar_Cl_1/ --phase train_target --load ./shot_Ar_1/checkpoints/best.pth --trade-off 0

# python shot.py /data/office-home/ --domain Ar --wn --val-ratio 0.1 --train-resizing ran.crop --epochs 50 --log ./shot_Ar_2/ --phase train_source --lb-smooth 0.1
# python shot.py /data/office-home/ --domain Cl --wn --val-ratio 0   --train-resizing ran.crop --epochs 15 --log ./shot_Ar_Cl_2/ --phase train_target --load ./shot_Ar_2/checkpoints/best.pth --trade-off 0.3
# python shot.py /data/office-home/ --domain Cl --wn --val-ratio 0   --train-resizing ran.crop --epochs 15 --log ./shotim_Ar_Cl_2/ --phase train_target --load ./shot_Ar_2/checkpoints/best.pth --trade-off 0

# python shot.py /data/office-home/ --domain Ar --wn --val-ratio 0.1 --train-resizing ran.crop --epochs 50 --log ./shot_Ar_3/ --phase train_source --lb-smooth 0.1
# python shot.py /data/office-home/ --domain Cl --wn --val-ratio 0   --train-resizing ran.crop --epochs 15 --log ./shot_Ar_Cl_3/ --phase train_target --load ./shot_Ar_3/checkpoints/best.pth --trade-off 0.3
# python shot.py /data/office-home/ --domain Cl --wn --val-ratio 0   --train-resizing ran.crop --epochs 15 --log ./shotim_Ar_Cl_3/ --phase train_target --load ./shot_Ar_3/checkpoints/best.pth --trade-off 0

# python shot.py /data/office-home/ --domain Ar --wn --val-ratio 0.1 --train-resizing ran.crop --epochs 50 --log ./shot_Ar_4/ --phase train_source --lb-smooth 0.1
# python shot.py /data/office-home/ --domain Cl --wn --val-ratio 0   --train-resizing ran.crop --epochs 15 --log ./shot_Ar_Cl_4/ --phase train_target --load ./shot_Ar_4/checkpoints/best.pth --trade-off 0.3
# python shot.py /data/office-home/ --domain Cl --wn --val-ratio 0   --train-resizing ran.crop --epochs 15 --log ./shotim_Ar_Cl_4/ --phase train_target --load ./shot_Ar_4/checkpoints/best.pth --trade-off 0

