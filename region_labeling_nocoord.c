/*
 * TP Final - Introduccion a la Concurrencia (UNPSJB)
 * Tema 4: Procesamiento de imagenes - Etiquetado de regiones (MPI sin coordinador)
 *
 * Comentarios:
 * Version paralela descentralizada en MPI (sin proceso coordinador), segun Andrews (pp. 434-435).
 * - La grilla N x N se divide en franjas horizontales de N / P filas por proceso, con dos filas
 *   halo (arriba y abajo) intercambiadas mediante MPI_Sendrecv en cada iteracion.
 * - A diferencia de la version con coordinador, aca nadie centraliza la decision de corte:
 *   todos los procesos llaman a MPI_Allreduce(..., MPI_LOR, ...) para saber al mismo tiempo
 *   si alguno tuvo cambios en esa vuelta.
 * - Solo al principio y al final se usa rank 0 para leer el archivo y mostrar el resumen/tiempo.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "mpi_comun.h"

int repartirFranjas(const char* archivo, int* n, int size, int** imagen_completa, int** labels_completas){
	*imagen_completa = leer_imagen(archivo, n);

	if (!(*imagen_completa))
        	return 0;

	/* Verificacion de procesos validos para imagen */
	if (*n % size != 0) {
        	printf("Error: Dimension N (%d) no es divisible por la cantidad de procesos (%d).\n", *n, size);
        	free(*imagen_completa);
        	*imagen_completa = NULL;
        	return 0;
	}

	/* Reservar memoria para etiquetas */
	*labels_completas = (int*)malloc((size_t)(*n) * (*n) * sizeof(int));
	if (!(*labels_completas)) {
        	perror("Error reservando memoria");
        	free(*imagen_completa);
        	*imagen_completa = NULL;
		return 0;
	}

	inicializar_etiquetas(*imagen_completa, *labels_completas, *n);
	return 1;
}

void inicializarVarsLocales(int n, int size, int* filas_locales, int* filas_con_halo, int* celdas_locales, int** local_grid, int** temp_grid, int rank, int* vecino_inferior, int* vecino_superior) {
	/* Cada proceso recibe N / P filas de la grilla */
	*filas_locales = n / size;

	/* Se agregan dos filas adicionales (halo) */
	*filas_con_halo = *filas_locales + 2;

	/* Cantidad de celdas que pertenecen realmente a cada proceso, sin contar los halos */
	*celdas_locales = (*filas_locales) * n;

	/* Reservacion de memoria local: local_grid: etiquetas actuales (con halo),
	   temp_grid: etiquetas actualizadas en una iteracion */
	*local_grid = (int*)calloc((size_t)(*filas_con_halo) * n, sizeof(int));
	*temp_grid = (int*)calloc((size_t)(*filas_con_halo) * n, sizeof(int));

	/* Verifica que ambos espacios de memoria se hayan reservado correctamente */
	if (!(*local_grid) || !(*temp_grid)) {
		perror("Error reservando memoria local");
		free(*local_grid);
		free(*temp_grid);
		*local_grid = NULL;
		*temp_grid = NULL;
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	/* Determinar los procesos vecinos (con los que se intercambiaran las filas halo) */
	*vecino_superior = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
	*vecino_inferior = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;
}

void printPieDePagina(int n, int size, int iteraciones, int regiones, double tiempo_total){
	printf("Dimension de grilla: %d x %d | Procesos: %d\n", n, n, size);
	printf("Iteraciones hasta converger: %d\n", iteraciones);
	printf("Regiones conexas encontradas: %d\n", regiones);
	printf("Tiempo de ejecucion MPI: %.6f seg\n", tiempo_total);
}

void printFinal(int n, int size, int* labels_completas, int* imagen_completa, int verbose, int iteraciones, double tiempo_total){
	int regiones = contar_regiones(labels_completas, n);

	if (verbose && n <= 32) {
		printf("Grilla final:\n");
		imprimir_grilla(labels_completas, n);
	}

	/* Presenta las estadisticas finales de la ejecucion */
	printPieDePagina(n , size, iteraciones, regiones, tiempo_total);

	free(imagen_completa);
	free(labels_completas);
}

int main(int argc, char* argv[]) {
	/* Inicializacion */
	int rank, size;
	int n = 0;
	int* imagen_completa = NULL;
	int* labels_completas = NULL;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	/* Verfica que la cantidad de los argumentos sea valida */
	if (argc < 2) {
		if (rank == 0) 
			printf("Uso: mpiexec -np <procesos> %s <archivo_imagen.txt> [-p] [-v]\n", argv[0]);
		
		MPI_Finalize();
		return 1;
	}

	/* Configura las opciones de ejecucion: p: muestra cada iteracion; v: muestra el final */
	int paso_a_paso = 0;
	int verbose = 0;
	parsear_flags(argc, argv, &paso_a_paso, &verbose);

	/* Leer imagen y repartir franjas */
	if (rank == 0) {
		if (!repartirFranjas(argv[1], &n, size, &imagen_completa, &labels_completas)){
			MPI_Abort(MPI_COMM_WORLD, 1);
			return 1;
		}else if(paso_a_paso && n <= 32){
			printf("Etiquetas iniciales:\n");
			imprimir_grilla(labels_completas, n);
		}
	}

	/* Distribucion de la dimension de la imagen */
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	/* Inicializacion de la memoria de franja propia */
	int filas_locales, filas_con_halo, celdas_locales, *local_grid, *temp_grid, vecino_inferior, vecino_superior;
	inicializarVarsLocales(n, size, &filas_locales, &filas_con_halo, &celdas_locales, &local_grid, &temp_grid, rank, &vecino_inferior, &vecino_superior);

	/* Distribucion de la grilla inicial entre los procesos: reciben sus filas en local_grid (comenzando en la fila 1) */
	MPI_Scatter(labels_completas, celdas_locales, MPI_INT, &local_grid[1 * n], celdas_locales, MPI_INT, 0, MPI_COMM_WORLD);

	/* Sincroniza los procesos antes de medir el tiempo */
	MPI_Barrier(MPI_COMM_WORLD);
	double t_inicio = MPI_Wtime();

	int iteraciones = 0;
	int cambio_global = 1;

	/* Deciden conjuntamente cuando finalizar, utilizando MPI_Allreduce */
	while (cambio_global) {
		iteraciones++;

		intercambiar_halos(local_grid, filas_locales, n, vecino_superior, vecino_inferior);

		/* 1: si hubo cambio en alguna etiqueta */
		int cambio_local = actualizar_etiquetas_locales(local_grid, temp_grid, filas_locales, n);

		/* Intercambio de punteros, para rotarlos */
		int* aux = local_grid;
		local_grid = temp_grid;
		temp_grid = aux;

		/* Actualizar el cambio global */
		MPI_Allreduce(&cambio_local, &cambio_global, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

		if (paso_a_paso && n <= 32) {
			MPI_Gather(&local_grid[1 * n], celdas_locales, MPI_INT, labels_completas, celdas_locales, MPI_INT, 0, MPI_COMM_WORLD);
			if (rank == 0) {
				printf("Iteracion %d:\n", iteraciones);
				imprimir_grilla(labels_completas, n);
			}
		}
	}

	/* Reconstruccion de la grilla completa */
	MPI_Gather(&local_grid[1 * n], celdas_locales, MPI_INT, labels_completas, celdas_locales, MPI_INT, 0, MPI_COMM_WORLD);

	/* Sincronizacion para medicion de tiempo */
	MPI_Barrier(MPI_COMM_WORLD);
	double t_fin = MPI_Wtime();
	double tiempo_total = t_fin - t_inicio;

	/* contar las regiones, mostrar los resultados y liberar la memoria global */
	if (rank == 0) 
		printFinal(n, size, labels_completas, imagen_completa, verbose, iteraciones, tiempo_total);

	free(local_grid);
	free(temp_grid);
	MPI_Finalize();
	return 0;
}