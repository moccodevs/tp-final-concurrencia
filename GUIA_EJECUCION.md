# Guia de ejecucion

## Requisitos

- gcc y make
- OpenMPI (mpicc y mpiexec)

## Compilar

```
make clean
make all
```

Genera tres programas:

- `region_labeling_seq`: version secuencial
- `region_labeling_coord`: MPI con coordinador
- `region_labeling_nocoord`: MPI sin coordinador

## Ejecutar

Secuencial:

```
./region_labeling_seq imagenes/demo_8.txt
```

MPI con coordinador:

```
mpiexec --oversubscribe -np 5 ./region_labeling_coord imagenes/demo_8.txt -p
```

MPI sin coordinador:

```
mpiexec --oversubscribe -np 4 ./region_labeling_nocoord imagenes/demo_8.txt -p
```

## Flags

- `-v`: imprime la grilla final (solo si N <= 32)
- `-p`: imprime la grilla en cada iteracion (solo si N <= 32)

## Cantidad de procesos

- Con coordinador: N debe ser divisible por np - 1 (np - 1 trabajadores).
  Ejemplo: N = 256 sirve con np = 2, 3, 5, 9.
- Sin coordinador: N debe ser divisible por np.

## Imagenes

La carpeta `imagenes/` tiene los archivos de prueba.
Formato: primera linea el tamano N, luego N*N valores 0 o 1.

## Pruebas de tiempo

```
./benchmark.sh
```

## Ejecucion en red

```
./red_mpi.sh preparar
./red_mpi.sh conectar usuario@IP
./red_mpi.sh correr nocoord imagenes/bloques_128.txt
```
