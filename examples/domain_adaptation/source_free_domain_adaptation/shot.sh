#!/usr/bin/env bash
# ResNet50, OfficeHome, Ar2all
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Ar -a resnet50 --val-ratio 0.1 --epochs 100 --log logs/shot/Ar/ --phase train_source --lb-smooth 0.1
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Cl -a resnet50 --val-ratio 0   --epochs 15  --log logs/shot/Ar2Cl/ --phase train_target --load-pretrained-model logs/shot/Ar/checkpoints/best.pth --pseudo-label-trade-off 0.3
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Pr -a resnet50 --val-ratio 0   --epochs 15  --log logs/shot/Ar2Pr/ --phase train_target --load-pretrained-model logs/shot/Ar/checkpoints/best.pth --pseudo-label-trade-off 0.3
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Rw -a resnet50 --val-ratio 0   --epochs 15  --log logs/shot/Ar2Rw/ --phase train_target --load-pretrained-model logs/shot/Ar/checkpoints/best.pth --pseudo-label-trade-off 0.3
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Cl -a resnet50 --val-ratio 0   --epochs 15  --log logs/shot/Ar2Cl_im/ --phase train_target --load-pretrained-model logs/shot/Ar/checkpoints/best.pth --pseudo-label-trade-off 0
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Pr -a resnet50 --val-ratio 0   --epochs 15  --log logs/shot/Ar2Pr_im/ --phase train_target --load-pretrained-model logs/shot/Ar/checkpoints/best.pth --pseudo-label-trade-off 0
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Rw -a resnet50 --val-ratio 0   --epochs 15  --log logs/shot/Ar2Rw_im/ --phase train_target --load-pretrained-model logs/shot/Ar/checkpoints/best.pth --pseudo-label-trade-off 0

# ResNet50, OfficeHome, Cl2all
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Cl -a resnet50 --val-ratio 0.1 --epochs 100 --log logs/shot/Cl/ --phase train_source --lb-smooth 0.1
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Ar -a resnet50 --val-ratio 0   --epochs 15  --log logs/shot/Cl2Ar/ --phase train_target --load-pretrained-model logs/shot/Cl/checkpoints/best.pth --pseudo-label-trade-off 0.3
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Pr -a resnet50 --val-ratio 0   --epochs 15  --log logs/shot/Cl2Pr/ --phase train_target --load-pretrained-model logs/shot/Cl/checkpoints/best.pth --pseudo-label-trade-off 0.3
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Rw -a resnet50 --val-ratio 0   --epochs 15  --log logs/shot/Cl2Rw/ --phase train_target --load-pretrained-model logs/shot/Cl/checkpoints/best.pth --pseudo-label-trade-off 0.3
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Ar -a resnet50 --val-ratio 0   --epochs 15  --log logs/shot/Cl2Ar_im/ --phase train_target --load-pretrained-model logs/shot/Cl/checkpoints/best.pth --pseudo-label-trade-off 0
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Pr -a resnet50 --val-ratio 0   --epochs 15  --log logs/shot/Cl2Pr_im/ --phase train_target --load-pretrained-model logs/shot/Cl/checkpoints/best.pth --pseudo-label-trade-off 0
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Rw -a resnet50 --val-ratio 0   --epochs 15  --log logs/shot/Cl2Rw_im/ --phase train_target --load-pretrained-model logs/shot/Cl/checkpoints/best.pth --pseudo-label-trade-off 0

# ResNet50, OfficeHome, Pr2all
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Pr -a resnet50 --val-ratio 0.1 --epochs 100 --log logs/shot/Pr/ --phase train_source --lb-smooth 0.1
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Ar -a resnet50 --val-ratio 0   --epochs 15  --log logs/shot/Pr2Ar/ --phase train_target --load-pretrained-model logs/shot/Pr/checkpoints/best.pth --pseudo-label-trade-off 0.3
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Cl -a resnet50 --val-ratio 0   --epochs 15  --log logs/shot/Pr2Cl/ --phase train_target --load-pretrained-model logs/shot/Pr/checkpoints/best.pth --pseudo-label-trade-off 0.3
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Rw -a resnet50 --val-ratio 0   --epochs 15  --log logs/shot/Pr2Rw/ --phase train_target --load-pretrained-model logs/shot/Pr/checkpoints/best.pth --pseudo-label-trade-off 0.3
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Ar -a resnet50 --val-ratio 0   --epochs 15  --log logs/shot/Pr2Ar_im/ --phase train_target --load-pretrained-model logs/shot/Pr/checkpoints/best.pth --pseudo-label-trade-off 0
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Cl -a resnet50 --val-ratio 0   --epochs 15  --log logs/shot/Pr2Cl_im/ --phase train_target --load-pretrained-model logs/shot/Pr/checkpoints/best.pth --pseudo-label-trade-off 0
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Rw -a resnet50 --val-ratio 0   --epochs 15  --log logs/shot/Pr2Rw_im/ --phase train_target --load-pretrained-model logs/shot/Pr/checkpoints/best.pth --pseudo-label-trade-off 0

# ResNet50, OfficeHome, Rw2all
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Rw -a resnet50 --val-ratio 0.1 --epochs 100 --log logs/shot/Rw/ --phase train_source --lb-smooth 0.1
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Ar -a resnet50 --val-ratio 0   --epochs 15  --log logs/shot/Rw2Ar/ --phase train_target --load-pretrained-model logs/shot/Rw/checkpoints/best.pth --pseudo-label-trade-off 0.3
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Cl -a resnet50 --val-ratio 0   --epochs 15  --log logs/shot/Rw2Cl/ --phase train_target --load-pretrained-model logs/shot/Rw/checkpoints/best.pth --pseudo-label-trade-off 0.3
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Pr -a resnet50 --val-ratio 0   --epochs 15  --log logs/shot/Rw2Pr/ --phase train_target --load-pretrained-model logs/shot/Rw/checkpoints/best.pth --pseudo-label-trade-off 0.3
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Ar -a resnet50 --val-ratio 0   --epochs 15  --log logs/shot/Rw2Ar_im/ --phase train_target --load-pretrained-model logs/shot/Rw/checkpoints/best.pth --pseudo-label-trade-off 0
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Cl -a resnet50 --val-ratio 0   --epochs 15  --log logs/shot/Rw2Cl_im/ --phase train_target --load-pretrained-model logs/shot/Rw/checkpoints/best.pth --pseudo-label-trade-off 0
CUDA_VISIBLE_DEVICES=0 python shot.py data/office-home/ -d OfficeHome --domain Pr -a resnet50 --val-ratio 0   --epochs 15  --log logs/shot/Rw2Pr_im/ --phase train_target --load-pretrained-model logs/shot/Rw/checkpoints/best.pth --pseudo-label-trade-off 0
