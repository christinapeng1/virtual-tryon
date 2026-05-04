# oscar/files/

This directory holds the SURREAL file lists used by `download_surreal.sh`.

These `.txt` files are provided by the SURREAL repository at:
https://github.com/gulvarol/surreal/tree/master/download/files

After cloning this repo on OSCAR, copy the file lists from the SURREAL repo here:

```bash
# On OSCAR, after cloning surreal separately:
cp /path/to/surreal/download/files/files_cmu_train.mp4.txt oscar/files/
cp /path/to/surreal/download/files/files_cmu_train_info.mat.txt oscar/files/
cp /path/to/surreal/download/files/files_cmu_val.mp4.txt oscar/files/
cp /path/to/surreal/download/files/files_cmu_val_info.mat.txt oscar/files/
```
