#/bin/bash

wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1cNs7VOeftHNpjjUBIy76gN37fST-5sSB' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1cNs7VOeftHNpjjUBIy76gN37fST-5sSB" -O siamese_cnn_model_cpu.pt && rm -rf ~/cookies.txt
