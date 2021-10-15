#!/bin/bash

#module load oneapi  
#export LIBOMPTARGET_PLUGIN=LEVEL0              # for OpenMP codes
#export SYCL_DEVICE_FILTER=level_zero            # for DPCPP codes

#OCCA_DIR=/path/to/occa
#export PRJ="/path/to/project/"
#export SRC1="/paths/to/source/"
#export SRC2="/paths/to/source/"
#export SRC3="$OCCA_DIR/lib/"

#Cache Directories
#export SRC4="/paths/to/occa/cache/dirs/"

# run
#RUNCOMM="./axhelm 7 1 8000 NATIVE+DPCPP DPCPP"
#ZE_AFFINITY_MASK=0.0 advisor --collect=roofline --profile-gpu --target-gpu=0:179:0.0 \
#                             --project-dir=$PRJ --search-dir all:r=$SRC1,$SRC2,$SRC3,$SRC4 -- $RUNCOMM 

#
#advisor --report=roofline --gpu --project-dir=$PRJ --report-output=${PRJ}/roofline.html
