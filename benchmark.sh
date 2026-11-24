#!/bin/bash
# Pruebas de tiempo: secuencial vs MPI.
# Se comparan igual cantidad de trabajadores:
#   nocoord recibe np = W procesos y usa los W.
#   coord recibe np = W + 1 procesos (1 coordinador + W trabajadores).

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

make all

IMAGENES=(
    imagenes/bloques_64.txt
    imagenes/bloques_128.txt
    imagenes/bloques_256.txt
)

for IMG in "${IMAGENES[@]}"; do
    echo "== $IMG =="
    ./region_labeling_seq "$IMG"
    for W in 2 4; do
        echo "-- $W trabajadores --"
        mpiexec --oversubscribe -np "$W" ./region_labeling_nocoord "$IMG"
        mpiexec --oversubscribe -np "$((W + 1))" ./region_labeling_coord "$IMG"
    done
done
