# Índice de experimentos

Esta es la nomenclatura canónica del proyecto. Los nombres históricos se conservan en `resultados/catalogo_ejecuciones.csv` para no romper la trazabilidad.

| Orden | Bloque | Experimento | Pregunta |
|---:|:---:|---|---|
| 1 | A | Transferencia del encoder | ¿Aportan los pesos preentrenados de OWSM frente a la inicialización aleatoria? |
| 2 | C | Estabilidad del entrenamiento | ¿Se mantiene el hallazgo entre semillas y ajustes de optimización? |
| 3 | D | Estrategia de adaptación | ¿Conviene congelar, usar LoRA o ajustar parcialmente el encoder? |
| 4 | E | Frontend de EEG | ¿Qué capacidad necesita la transformación de entrada? |
| 5 | F | Profundidad del encoder | ¿Cuántos bloques E-Branchformer son necesarios? |
| 6 | G | Unidad de salida CTC | ¿Qué granularidad de salida minimiza el WER textual? |
| 7 | H | Generalización entre sesiones | ¿Qué ocurre al evaluar sesiones ausentes del entrenamiento? |
| 8 | I | Modelo de lenguaje externo | ¿Cuánto mejora KenLM la búsqueda CTC? |
| 9 | J | Decoder OWSM preliminar | ¿Qué ocurrió en los primeros ensayos con el decoder de atención? |
| 10 | B | Sistema final | ¿Cuál es la mejor configuración completa y cuál es su WER final? |

`P` agrupa únicamente pruebas preliminares, fallos y versiones de desarrollo; no es un bloque de resultados de la memoria.

## Notebooks principales

| Notebook | Función |
|---|---|
| `G01_objetivo_caracter.ipynb` | Receta final del objetivo CTC de caracteres. |
| `G02_objetivo_fonema.ipynb` | Receta final del objetivo CTC de fonemas. |
| `B01_sistema_final_palabra.ipynb` | Receta del sistema campeón; bloque todavía abierto. |
| `J01_decoder_owsm_preliminar.ipynb` | Adaptación inicial con decoder de atención OWSM. |
| `analisis_resultados_tfg.ipynb` | Generación unificada de figuras, tablas y análisis. |
| `espnet_eeg.py` | Implementación compartida, no es un experimento. |

## Convención de artefactos

Las figuras y tablas se nombran como `<tipo>_<orden>_<bloque><número>_<descripción>`. Por ejemplo: `fig_01_A01_curvas_transferencia_encoder.pdf`.

Las carpetas crudas de resultados mantienen su `exp_tag` original. Renombrarlas rompería las referencias desde `resultados.csv`, los archivos por época y los resultados de KenLM.
