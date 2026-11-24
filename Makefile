# Makefile para TP Final - Introduccion a la Concurrencia
# Tema: Procesamiento de Imagenes: Etiquetado de Regiones
# UNPSJB - Sede Trelew

CC = mpicc
GCC = gcc
CFLAGS = -Wall -Wextra -O2
LDFLAGS = -lm

TARGETS = region_labeling_seq region_labeling_coord region_labeling_nocoord

all: $(TARGETS)

region_labeling_seq: region_labeling_seq.c imagen_comun.h
	$(GCC) $(CFLAGS) region_labeling_seq.c -o region_labeling_seq $(LDFLAGS)

region_labeling_coord: region_labeling_coord.c mpi_comun.h imagen_comun.h
	$(CC) $(CFLAGS) region_labeling_coord.c -o region_labeling_coord $(LDFLAGS)

region_labeling_nocoord: region_labeling_nocoord.c mpi_comun.h imagen_comun.h
	$(CC) $(CFLAGS) region_labeling_nocoord.c -o region_labeling_nocoord $(LDFLAGS)

clean:
	rm -f $(TARGETS) *.o
