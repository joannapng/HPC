TARGET=$1
SIZE=$2
DIR="../Code/"
FILE="results_"
FILENAME="$DIR""$FILE""$SIZE"".txt"

export OMP_NUM_THREADS=64

if [[ -f "$FILENAME" ]]; then
    ./$TARGET $SIZE    
else
    CUR_DIR=$(pwd)
    cd ../Code
    make clean
    make cpu
    ./nbody_cpu $SIZE
    cd $CUR_DIR
    ./$TARGET $SIZE
fi
