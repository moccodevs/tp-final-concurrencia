/*
* TP Final - Introduccion a la Concurrencia (UNPSJB)
* Tema 4: Procesamiento de imagenes - Etiquetado de regiones (MPI con coordinador)
*
* Comentarios:
* Version paralela en MPI usando un proceso coordinador (rank 0), segun Andrews (pp. 432-434).
* - La grilla N x N se reparte por franjas horizontales entre los P procesos (N / P filas para c/u de los trabajadores).
* - Cada trabajador reserva (filas_locales + 2) * N enteros para tener dos filas extra (halos):
*   la fila 0 guarda la fila borde del vecino de arriba y la fila (filas_locales + 1) la del de abajo.
* - En cada iteracion se intercambian halos con MPI_Sendrecv para evitar deadlocks.
* - Despues cada proceso actualiza sus celdas y manda si tuvo cambios (0 o 1) al rank 0
*   con MPI_Reduce. El coordinador suma los cambios y avisa por MPI_Bcast si siguen o terminan.
* - Al finalizar, rank 0 junta todas las franjas con MPI_Gather.
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "mpi_comun.h"

void printPieDePagina(int n, int size, int trabajadores, int iteraciones, int regiones, double tiempo_total){
    printf("Dimension de grilla: %d x %d\n", n, n);
    printf("Procesos MPI: %d (1 coordinador + %d trabajadores)\n", size, trabajadores);
    printf("Iteraciones hasta converger: %d\n", iteraciones);
    printf("Regiones conexas encontradas: %d\n", regiones);
    printf("Tiempo de ejecucion MPI: %.6f seg\n", tiempo_total);
}

void inicializarCoordinador(char *argv[], int size, int *n_out, int **imagen_completa_out, int **labels_completas_out) {
    int n = 0;
    int *imagen_completa = NULL;
    int *labels_completas = NULL;
    const int trabajadores = size - 1;

    /* Inicializacion de la imagen */
    imagen_completa = leer_imagen(argv[1], &n);
    if (imagen_completa == NULL) {
        fprintf(stderr, "Error al leer la imagen.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Verificacion de la dimension */
    if (n % trabajadores != 0) {
        fprintf(stderr, "Error: Dimension N (%d) no es divisible por la cantidad de trabajadores (%d).\n", n, trabajadores);
        free(imagen_completa);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Inicializacion de etiquetas */
    labels_completas = (int *)malloc((size_t)n * n * sizeof(int));
    if (labels_completas == NULL) {
        perror("Error reservando memoria en coordinador");
        free(imagen_completa);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    inicializar_etiquetas(imagen_completa, labels_completas, n);

    /* Asignar los valores a los punteros de salida */
    *n_out = n;
    *imagen_completa_out = imagen_completa;
    *labels_completas_out = labels_completas;
}

void coordinador(char *argv[], int size, int paso_a_paso, int verbose) {

    int n = 0;
    int *imagen_completa = NULL;
    int *labels_completas = NULL;

    inicializarCoordinador(argv, size, &n, &imagen_completa, &labels_completas);

    const int trabajadores = size - 1;

    /* --------------------------------------------------------
     * Distribucion de la imagen
     * -------------------------------------------------------- */

    /* Comunica la dimension a todos los trabajadores. */
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const int filas_locales = n / trabajadores;
    const int celdas_locales = filas_locales * n;

    /*
     * counts[r]: cantidad de elementos que recibe/envia el rank r.
     * displs[r]: desplazamiento dentro de la matriz completa.
     * El rank 0 es coordinador y no recibe ninguna franja.
     */
    int *counts = (int *)calloc((size_t)size, sizeof(int));
    int *displs = (int *)calloc((size_t)size, sizeof(int));

    if (counts == NULL || displs == NULL) {
        perror("Error reservando memoria para Scatterv/Gatherv");
        free(counts);
        free(displs);
        free(imagen_completa);
        free(labels_completas);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Rank 0 mantiene counts[0] = 0. */
    for (int r = 1; r < size; r++) {
        counts[r] = celdas_locales;
        displs[r] = (r - 1) * celdas_locales;
    }

    /* Buffer auxiliar: el coordinador no recibe ni envia celdas. */
    int dummy = 0;

    MPI_Scatterv(labels_completas, counts, displs, MPI_INT, &dummy, 0, MPI_INT, 0, MPI_COMM_WORLD);

    /* Inicio del procesamiento y medicion del tiempo */
    MPI_Barrier(MPI_COMM_WORLD);
    const double t_inicio = MPI_Wtime();

    int iteraciones = 0;
    int continuar = 1;

    /* Protocolo de convergencia */
    while (continuar) {
        iteraciones++;

        /* El coordinador aporta 0 porque no actualiza celdas. */
        int cambio_local = 0;
        int cambio_global = 0;

        /* Recibe la suma de los cambios informados por workers. */
        MPI_Reduce(&cambio_local, &cambio_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        /* Si hubo cambios, se ejecuta otra iteracion. */
        continuar = (cambio_global > 0) ? 1 : 0;
        MPI_Bcast(&continuar, 1, MPI_INT, 0, MPI_COMM_WORLD);

        /* Visualizacion paso a paso */
        if (paso_a_paso && n <= 32) {
            MPI_Gatherv(&dummy, 0, MPI_INT, labels_completas, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

            printf("Iteracion %d:\n", iteraciones);
            imprimir_grilla(labels_completas, n);
            fflush(stdout);
        }
    }

    /* Recoleccion de la matriz final */
    MPI_Gatherv(&dummy, 0, MPI_INT, labels_completas, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    /* Sincroniza antes de detener el cronometro. */
    MPI_Barrier(MPI_COMM_WORLD);
    const double t_fin = MPI_Wtime();
    const double tiempo_total = t_fin - t_inicio;

    /* Conteo de regiones y presentacion de resultados */
    const int regiones = contar_regiones(labels_completas, n);

    if (verbose && n <= 32 && !paso_a_paso) {
        printf("Grilla final:\n");
        imprimir_grilla(labels_completas, n);
    }

    printPieDePagina(n, size, trabajadores, iteraciones, regiones, tiempo_total);

    /* Liberacion de memoria del coordinador */
    free(counts);
    free(displs);
    free(imagen_completa);
    free(labels_completas);
}

void trabajador(int rank, int size, int paso_a_paso) {

    int n = 0;

    /* Recepcion de la dimension de la imagen */
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const int trabajadores = size - 1;
    const int filas_locales = n / trabajadores;
    const int filas_con_halo = filas_locales + 2;
    const int celdas_locales = filas_locales * n;

    /* Reserva de memoria local */
    int *local_grid = (int *)calloc((size_t)filas_con_halo * n, sizeof(int));
    int *temp_grid = (int *)calloc((size_t)filas_con_halo * n, sizeof(int));

    if (local_grid == NULL || temp_grid == NULL) {
        perror("Error reservando memoria local");
        free(local_grid);
        free(temp_grid);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Recepcion de la franja (La franja propia se almacena desde local_grid[n]. Filas 1 a filas_locales: datos propios.) */
    MPI_Scatterv(NULL, NULL, NULL, MPI_INT, &local_grid[n], celdas_locales, MPI_INT, 0, MPI_COMM_WORLD);

    /* Identificacion de vecinos */
    const int vecino_superior = (rank > 1) ? rank - 1 : MPI_PROC_NULL;
    const int vecino_inferior = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    /* Todos comienzan el procesamiento despues de la distribucion. */
    MPI_Barrier(MPI_COMM_WORLD);

    int continuar = 1;

    /* Procesamiento */
    while (continuar) {

        /* Intercambia halos solamente con los trabajadores vecinos. */
        intercambiar_halos(local_grid, filas_locales, n, vecino_superior, vecino_inferior);

        /* Actualiza las etiquetas de las celdas propias. */
        int cambio_local = actualizar_etiquetas_locales(local_grid, temp_grid, filas_locales, n);

        /* Intercambia los buffers para la siguiente iteracion. */
        int *aux = local_grid;
        local_grid = temp_grid;
        temp_grid = aux;

        /* Informa al coordinador si hubo cambios locales. */
        MPI_Reduce(&cambio_local, NULL, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        /* Recibe la decision global de continuar o finalizar. */
        MPI_Bcast(&continuar, 1, MPI_INT, 0, MPI_COMM_WORLD);

        /* Visualizacion paso a paso */
        if (paso_a_paso && n <= 32) {
            MPI_Gatherv(&local_grid[n], celdas_locales, MPI_INT, NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }

    /* Recoleccion de la matriz final */
    MPI_Gatherv(&local_grid[n], celdas_locales, MPI_INT, NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);

    /* Debe coincidir con la barrera final del coordinador. */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Liberacion de memoria del trabajador */
    free(local_grid);
    free(temp_grid);
}

int verificacionDeArgumentos(int argc, char *argv[], int size, int rank){
    if (argc < 2) {
        if (rank == 0)
            printf("Uso: mpiexec -np <procesos> %s <archivo_imagen.txt> [-p] [-v]\n", argv[0]);
        return 0;
    }

    /* Se necesita un coordinador y al menos un trabajador. */
    if (size < 2) {
        if (rank == 0)
            printf("Error: Se necesita un coordinador y al menos un trabajador.\n");
        return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {

    int rank = 0;
    int size = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Verificacion de validez de argumentos */
    if (!verificacionDeArgumentos(argc, argv, size, rank)){
        MPI_Finalize();
        return 1;
    }

    /* Lectura de parametros opcionales */
    int paso_a_paso = 0;
    int verbose = 0;

    parsear_flags(argc, argv, &paso_a_paso, &verbose);

    /* Asignacion de responsabilidades */
    if (rank == 0) {
        coordinador(argv, size, paso_a_paso, verbose);
    } else {
        trabajador(rank, size, paso_a_paso);
    }

    MPI_Finalize();
    return 0;
}