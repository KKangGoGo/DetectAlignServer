#/bin/bash

wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_cxE81Y30aGx6QTF3MMqRbiXDJArYvbl' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_cxE81Y30aGx6QTF3MMqRbiXDJArYvbl" -O siamese_cnn_model_gpu.pt && rm -rf ~/cookies.txt
