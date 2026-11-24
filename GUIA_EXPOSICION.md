# Guía de Exposición — Etiquetado de Regiones con MPI

Guía para presentar el TP Final de **Introducción a la Concurrencia** (UNPSJB).
Incluye: explicación técnica, guion palabra por palabra y resumen de una carilla.

---

## 1. Qué resuelve (el problema)

Dada una imagen binaria `N × N` (`0` = fondo, `1` = encendido), asignar **la misma etiqueta a todos los píxeles `1` conectados** entre sí, y etiquetas distintas a grupos separados.

Se usa **4-conectividad**: dos píxeles `1` están conectados si son vecinos ortogonales (arriba, abajo, izquierda, derecha).

```
V₄(i, j) = { (i-1, j), (i+1, j), (i, j-1), (i, j+1) }
```

Es *Connected Component Labeling*. En `imagenes/demo_8.txt` hay 4 regiones y el resultado final las marca como `19`, `23`, `52`, `55` (el número es la etiqueta máxima de cada isla).

---

## 2. El algoritmo: "latido" (heartbeat)

Base: Andrews, Cap. 9, pp. 430–435. Tiene 3 pasos:

1. **Inicialización** (`imagen_comun.h:56`): cada `0` queda en `0`; cada `1` recibe una etiqueta única según su posición:

   ```
   label[i][j] = i * N + j + 1   si imagen[i][j] = 1
   label[i][j] = 0               si imagen[i][j] = 0
   ```

2. **Iteración** (`region_labeling_seq.c:23`): cada celda encendida compara su etiqueta con sus 4 vecinos y **se queda con la mayor**:

   ```
   label[i][j] = max( label[i][j], vecinos )
   ```

   Como las etiquetas son únicas por posición, el máximo de la isla se propaga de a un vecino por vuelta, como una onda (de ahí "latido").

3. **Parada**: cuando una vuelta completa no cambia ninguna celda. En ese momento el máximo de cada componente llegó a todos sus píxeles.

**Detalle clave:** se usan **dos buffers** (`cur` y `next`) que se alternan (`swap`). Así se lee del estado viejo y se escribe el nuevo sin pisar datos a medio actualizar (actualización síncrona).

---

## 3. Los archivos

| Archivo | Rol |
|---|---|
| `imagen_comun.h` | Utilidades compartidas: leer imagen, inicializar etiquetas, imprimir, contar regiones, parsear flags |
| `mpi_comun.h` | Utilidades MPI: intercambio de halos y actualización local |
| `region_labeling_seq.c` | Versión secuencial (1 proceso). Base de verdad y de tiempo `T_seq` |
| `region_labeling_coord.c` | MPI **con** coordinador (maestro/trabajador) |
| `region_labeling_nocoord.c` | MPI **sin** coordinador (descentralizado) |
| `generador_imagenes.c` | Genera patrones de prueba (bloques, tablero, espiral, ruido) |
| `benchmark.sh` | Corre secuencial vs MPI con varios `N` y `P` |

---

## 4. Cómo se paraleliza: franjas + halos

La matriz se parte en **franjas horizontales**. Con `P` procesos, cada uno recibe `B = N/P` filas.
**Condición obligatoria: `N` divisible por la cantidad de procesos** (o trabajadores).

El problema: para actualizar su primera y última fila, un proceso necesita la fila vecina que está en otro proceso. Solución: **filas fantasma (halos)**. Cada proceso reserva `(B+2) × N`:

```
fila 0        -> halo superior (copia del borde del vecino de arriba)
filas 1..B    -> datos propios
fila B+1      -> halo inferior (copia del borde del vecino de abajo)
```

En cada iteración se intercambian esos bordes con `MPI_Sendrecv` (`mpi_comun.h:8`):

```c
MPI_Sendrecv(&grid[1*n],      n, MPI_INT, vecino_arriba, 0,
             &grid[0*n],      n, MPI_INT, vecino_arriba, 0, ...);
MPI_Sendrecv(&grid[B*n],      n, MPI_INT, vecino_abajo,  0,
             &grid[(B+1)*n],  n, MPI_INT, vecino_abajo,  0, ...);
```

Puntos clave para defender:

- **`Sendrecv`** hace envío y recepción en **una sola llamada** y evita **deadlock** (si ambos vecinos hicieran `Send` bloqueante a la vez, se quedarían esperando).
- En los bordes reales de la imagen se pasa `MPI_PROC_NULL`, que no envía ni recibe nada (no hay vecino).
- Solo se comunican `O(N)` enteros por iteración, mientras se calculan `O(N²/P)` celdas → buena relación cómputo/comunicación al crecer `N`.

---

## 5. Versión con coordinador (`region_labeling_coord.c`)

Esquema **maestro/trabajador**: `rank 0` = coordinador, `rank 1..P-1` = trabajadores.

1. El coordinador lee el archivo, inicializa etiquetas (`inicializarCoordinador`, línea 69) y verifica `N % (P-1) == 0`.
2. Reparte franjas con **`MPI_Scatterv`** (`distribucionImagen:30`): `counts[r]` y `displs[r]` indican cuántos elementos y desde dónde recibe cada worker; el coordinador recibe 0.
3. Bucle (línea 132):
   - El worker intercambia halos y actualiza (`trabajador:217`).
   - Cada worker manda su bandera `cambio_local` (0/1) con **`MPI_Reduce(..., MPI_SUM, 0, ...)`**.
   - El coordinador suma, decide `continuar = (cambio_global > 0)` y lo comunica con **`MPI_Bcast`**.
4. Al final junta todo con **`MPI_Gatherv`** y `rank 0` cuenta regiones e imprime.

Costo por iteración: **dos colectivas** (`Reduce` + `Bcast`), siempre pasando por el rank 0.
Ventaja: lógica centralizada y fácil de seguir. Desventaja: el coordinador **no calcula**, se desperdicia un proceso (P-1 trabajadores) y hay un cuello de botella en rank 0.

---

## 6. Versión sin coordinador (`region_labeling_nocoord.c`)

Esquema **descentralizado**: **todos** los procesos calculan (no hay rank 0 ocioso durante el cómputo).

- `rank 0` solo lee el archivo y reparte al inicio; luego también trabaja como uno más.
- Reparto con **`MPI_Scatter`** (bloques iguales, más simple que `Scatterv`).
- El corte se decide con **una sola colectiva**:

  ```c
  MPI_Allreduce(&cambio_local, &cambio_global, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
  ```

  `MPI_LOR` = OR lógico distribuido: si **algún** proceso cambió, `cambio_global = 1` para **todos** al mismo tiempo. No hace falta `Bcast`.
- Al final, `MPI_Gather` reconstruye la matriz y `rank 0` imprime.

Costo: **una colectiva** por iteración (`Allreduce`, `O(log P)`), en vez de `Reduce`+`Bcast`. Además aprovecha los P procesos completos.

---

## 7. Comparación (lo que seguro te preguntan)

- **Coordinador:** `Reduce` + `Bcast` = 2 colectivas/iteración + 1 proceso inactivo.
- **Sin coordinador:** `Allreduce` = 1 colectiva/iteración + todos calculan.
- **Justicia de la comparación:** con `-np 4`, el coordinador usa **3 trabajadores** y el nocoord usa **4**. Para comparar de igual a igual habría que correr el coordinador con `-np (P+1)`. Es un punto fino que conviene mencionar (`benchmark.sh` usa el mismo `-np` para ambos).
- **Métricas** (`INFORME_TECNICO.md:110`): `Speedup S_p = T_seq / T_p`, `Eficiencia E_p = S_p / P`.
- **Cuándo se nota la mejora:**
  - Imágenes chicas (`N=64`) → domina la comunicación.
  - Imágenes grandes (`N≥256`) → mejor proporción cómputo/comunicación.
  - Patrón **espiral** → muchas iteraciones (la etiqueta recorre toda la forma), donde el ahorro de `Allreduce` pesa más.
  - Patrón **bloques** → converge rápido en pocas vueltas.

### 7.1. Resultados medidos (equipo local, 24 núcleos)

| Imagen | Iteraciones | Secuencial | Coordinador | Sin coordinador |
|---|---|---|---|---|
| `bloques_64` | 15 | **0.000076 s** | 0.000084 s (np=3) | 0.000111 s (np=4) |
| `bloques_256` | 63 | 0.003917 s | 0.001710 s (np=5) | **0.001583 s** (np=4) |
| `espiral_512` | 1015 | 0.260806 s | 0.071517 s (np=5) | **0.060967 s** (np=4) |
| `espiral_1024` | 2039 | 1.985084 s | 0.513251 s (np=5) | **0.495279 s** (np=4) |

> Nota: para el coordinador se usó `np=5` (1 coordinador + **4 trabajadores**) para comparar contra los **4** procesos del nocoord. Con el mismo `np` el coordinador pierde todavía más, porque desperdicia un proceso.

**Lectura de los números:**

- **Imágenes chicas** (64×64, pocas iteraciones): gana la **secuencial**, porque el costo de comunicación supera al cómputo (MPI ≈ 0.000084–0.000111 s vs 0.000076 s).
- **Imágenes grandes**: ambas MPI le ganan claramente a la secuencial. En `espiral_1024` el speedup es ≈ **4×** (1.985 → ~0.50 s).
- **Entre coord y nocoord**: el **sin coordinador es levemente más rápido** (512: 0.0610 vs 0.0715; 1024: 0.495 vs 0.513), por hacer 1 colectiva en vez de 2 y no tener un proceso ocioso.
- **Requisito práctico**: el coordinador necesita `N % (np-1) == 0`, lo que complica elegir `np` (con `np=4` y `N=256` falla, porque 256 no es divisible por 3).

**Conclusión:** para este problema y tamaños razonables, la **versión sin coordinador es la más eficiente**; la secuencial solo conviene en imágenes muy chicas; la con coordinador es la más simple de razonar pero la menos eficiente de las dos paralelas.

---

## 8. Medición de tiempo

- Secuencial: `clock_gettime(CLOCK_MONOTONIC)` solo alrededor del etiquetado (`region_labeling_seq.c:116`).
- MPI: `MPI_Wtime()` entre `MPI_Barrier` inicial y final (`coord:125/160`, `nocoord:147/183`). La barrera asegura que todos arrancan y terminan juntos, para medir tiempo de pared real.
- Nota honesta: el secuencial no incluye lectura/init y el MPI incluye las colectivas finales; no es 100% "manzana con manzana", pero es la comparación estándar.

---

## 9. Guion palabra por palabra (10–12 min)

### Apertura (30 seg)

> "Buenos días. Nuestro trabajo es el etiquetado de regiones, o *Connected Component Labeling*, sobre imágenes binarias. Dada una matriz de ceros y unos, queremos que todos los unos conectados entre sí queden con el mismo número, y que cada grupo separado tenga un número distinto. Lo resolvimos con el algoritmo de latido del capítulo 9 de Andrews y lo paralelizamos con MPI en dos variantes: con coordinador y sin coordinador."

### Problema y ejemplo (1 min)

> "La imagen es una matriz cuadrada N por N, con 0 como fondo y 1 como píxel encendido. Usamos 4-conectividad: dos píxeles están conectados si son vecinos de arriba, abajo, izquierda o derecha. En el ejemplo de 8 por 8 que está en el repositorio hay cuatro regiones separadas, y el programa termina etiquetándolas como 19, 23, 52 y 55."

### El algoritmo (2 min)

> "El algoritmo tiene tres pasos. Primero, la inicialización: cada celda apagada queda en cero y cada celda encendida recibe una etiqueta única, que es su posición lineal, i por N más j más uno. Segundo, la iteración: cada celda encendida compara su etiqueta con las de sus cuatro vecinos y se queda con la mayor. Como las etiquetas son únicas, el máximo de cada isla se va propagando de a un vecino por vuelta, como una onda; de ahí el nombre de latido. Tercero, la parada: repetimos hasta que en una vuelta completa no cambie ninguna celda. En ese momento la etiqueta máxima de cada región llegó a todos sus píxeles. Un detalle de implementación importante: usamos dos buffers y los alternamos, para leer siempre del estado viejo y escribir el nuevo sin pisar datos a medio actualizar."

### Paralelización: franjas y halos (2 min)

> "Para repartir el trabajo con MPI, partimos la matriz en franjas horizontales: cada proceso recibe N sobre P filas. El problema es que, para actualizar su primera y última fila, un proceso necesita la fila del vecino, que está en otro proceso. Entonces cada proceso reserva dos filas extra, llamadas halos: la fila cero guarda el borde del proceso de arriba y la última el borde del de abajo. En cada iteración intercambiamos esos bordes con MPI_Sendrecv, que hace envío y recepción en una sola llamada y evita el deadlock. En los bordes reales de la imagen usamos MPI_PROC_NULL, que no envía ni recibe nada porque no hay vecino. Así solo comunicamos algo del orden de N enteros por vuelta, mientras calculamos N cuadrado sobre P celdas: la comunicación es barata comparada con el cómputo cuando la imagen crece."

### Con coordinador (2 min)

> "En la primera versión hay un proceso coordinador, el rank 0, y el resto son trabajadores. Al principio, el coordinador lee la imagen, inicializa las etiquetas y reparte las franjas con MPI_Scatterv. En cada iteración, cada trabajador intercambia halos, actualiza sus celdas y manda una bandera de cambio al coordinador con MPI_Reduce. El coordinador suma esas banderas, decide si hay que seguir, y avisa a todos con MPI_Bcast. Al final junta las franjas con MPI_Gatherv y cuenta las regiones. La desventaja es que el coordinador no calcula, así que con P procesos trabajamos realmente con P menos uno, y todas las decisiones pasan por un solo proceso."

### Sin coordinador (2 min)

> "La segunda versión es descentralizada: todos los procesos calculan, incluido el rank 0, que solo lee el archivo al principio. El reparto se hace con MPI_Scatter, que es más simple porque los bloques son iguales. La diferencia principal está en la decisión de corte: en lugar de Reduce más Bcast, todos los procesos llaman juntos a MPI_Allreduce con la operación OR lógico. Con un solo llamado, todos se enteran al mismo tiempo de si algún proceso tuvo cambios. Eso reduce dos colectivas por iteración a una, y aprovecha los P procesos completos. Al final se reconstruye la matriz con MPI_Gather."

### Comparación y cierre (1.5 min)

> "En resumen: la versión con coordinador es más fácil de leer y centraliza la lógica, pero desperdicia un proceso y hace dos comunicaciones colectivas por vuelta. La versión sin coordinador hace una sola y usa todos los procesos, a costa de una lógica más distribuida. Medimos tiempo de pared con MPI_Wtime entre barreras y calculamos speedup como tiempo secuencial sobre tiempo paralelo, y eficiencia como speedup sobre P. En imágenes chicas pesa más la comunicación; en imágenes grandes y en patrones como la espiral, que necesita muchas iteraciones, se nota más la ventaja de Allreduce. Como conclusión: para este problema, el esquema sin coordinador es el más eficiente porque elimina el cuello de botella y el proceso ocioso."

### Demo (si hay tiempo)

```bash
./region_labeling_seq imagenes/demo_8.txt -v
mpiexec --oversubscribe -np 4 ./region_labeling_coord imagenes/demo_8.txt -v
mpiexec --oversubscribe -np 4 ./region_labeling_nocoord imagenes/demo_8.txt -v
```

> "Como ven, las tres versiones dan exactamente las mismas cuatro regiones; la secuencial sirve para verificar que las paralelas son correctas."

---

## 10. Resumen de una carilla (para tener al lado)

**Problema:** etiquetar componentes conexas en imagen binaria N×N, 4-conectividad.
**Algoritmo (heartbeat):** inicializar etiqueta única `i*n+j+1`; iterar `label = max(self, 4 vecinos)`; parar cuando no hay cambios. Doble buffer para actualización síncrona.
**Paralelización:** franjas horizontales `B = N/P`; halos arriba/abajo; `MPI_Sendrecv` evita deadlock; `MPI_PROC_NULL` en bordes.
**Con coordinador (rank 0):** `Scatterv` reparte; por iteración `Reduce(SUM)` + `Bcast`; al final `Gatherv`. → 2 colectivas/vuelta, 1 proceso ocioso.
**Sin coordinador:** `Scatter`; por iteración `Allreduce(LOR)`; al final `Gather`. → 1 colectiva/vuelta, todos calculan.
**Métricas:** `S = T_seq / T_p`, `E = S / P`. `MPI_Wtime` entre `Barrier`.
**Conclusión:** nocoord es más eficiente (menos comunicación y sin proceso ocioso); coord es más simple de razonar.
**Requisito:** `N` divisible por la cantidad de procesos/trabajadores.

---

## 11. Preguntas probables y respuestas cortas

- **¿Por qué el máximo y no el mínimo?**
  Cualquiera sirve si es consistente; se eligió max porque la etiqueta inicial es la posición y el máximo de la isla es único y fácil de propagar.

- **¿Cuántas iteraciones tarda?**
  Tantas como el diámetro, en saltos, de la componente más grande. En espiral muchas; en bloques pocas.

- **¿Por qué dos buffers?**
  Para no leer datos ya modificados en la misma vuelta (actualización síncrona, tipo Jacobi).

- **¿Por qué `Sendrecv` y no `Send`/`Recv`?**
  Porque `Sendrecv` combina envío y recepción y evita que dos vecinos queden bloqueados esperándose mutuamente.

- **¿Qué es `MPI_PROC_NULL`?**
  Un "vecino nulo": las operaciones con él no envían ni reciben nada. Se usa en los bordes de la imagen.

- **¿Por qué `N` tiene que ser divisible?**
  Porque se reparten franjas iguales. Si no, habría que usar tamaños variables con `Scatterv`/`Gatherv` en ambas versiones.

- **¿El rank 0 del coordinador calcula?**
  No, solo coordina. Por eso se pierde un proceso y el reparto efectivo es entre `P-1` trabajadores.

- **¿Por qué `Allreduce` es mejor que `Reduce`+`Bcast`?**
  Porque resuelve en una sola colectiva `O(log P)` lo que la otra versión hace en dos, y no hace pasar todo por un proceso central.

- **¿Cómo verifican que está bien?**
  Comparando la salida de las dos versiones MPI contra la secuencial sobre las mismas imágenes; deben coincidir en regiones y en la grilla final.
