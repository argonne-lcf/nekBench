#!/bin/bash

#module load oneapi  
#export LIBOMPTARGET_PLUGIN=LEVEL0              # for OpenMP codes
export SYCL_DEVICE_FILTER=level_zero            # for DPCPP codes

OCCA_DIR=/home/spatel/A21/OCCA/OCCA_DPCPP/occa_9.21/occa/install_OAP_08.15.001/
export PRJ="/home/spatel/A21/CEED/nekBench/Advisor_Data/axhelm/"
export SRC1="/home/spatel/A21/CEED/nekBench/install/"
export SRC2="/home/spatel/A21/CEED/nekBench/install/lib/"
export SRC3="$OCCA_DIR/lib/"

#Cache Directories
export SRC4="/home/spatel/.occa/cache/c99d21435bd16115/"

# run
RUNCOMM="./axhelm 7 1 8000 NATIVE+DPCPP DPCPP"
#ZE_AFFINITY_MASK=0.0 advisor --collect=roofline --profile-gpu --target-gpu=0:179:0.0 \
#                             --project-dir=$PRJ --search-dir all:r=$SRC1,$SRC2,$SRC3,$SRC4 -- $RUNCOMM 

#
advisor --report=roofline --gpu --project-dir=$PRJ --report-output=${PRJ}/roofline.html
