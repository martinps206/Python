# Documentación técnica: Pipeline educativo de la subrutina de Shor

## Objetivo
Proveer una herramienta CLI y módulos auxiliares para:
- Simular la subrutina de búsqueda de periodo (IQFT + exponenciación modular)
- Ejecutar múltiples intentos variando 'a'
- Guardar resultados (JSON) y visualizar histogramas del registro de fase
- Servir como base pedagógica y de experimentación (añadir ruido, conectar Qiskit)

## Descripción del método
1. Preparación: registro de fase en superposición |x>, registro objetivo en |1>.
2. Aplicación de U_a: |x>|1> -> |x>|a^x mod N>.
3. IQFT sobre el registro de fase: aparicion de picos en fracciones ~ k/r.
4. Medición del registro de fase y uso de fracciones continuadas para estimar r.
5. Factorización: si r par y a^{r/2} != -1 (mod N), gcd(a^{r/2} ± 1, N) devuelve factor no trivial.

## Notas matemáticas y de complejidad
- Complejidad teórica (Shor ideal): poli en log N (puertas ~ O((log N)^3)), qubits ~ O(log N).
- En práctica: la corrección de errores incrementa los qubits físicos por qubit lógico notablemente.
- Esta simulación usa representación completa del vector (no escalable, but pedagogical).

## Limitaciones de esta implementación
- No modela ruido cuántico (NISQ) explícitamente.
- No implementa una versión optimizada de la exponenciación modular (se hace con amplitudes).
- Escalable solo para N pequeños (p.ej. 15, 21, 33) debido a memoria.

## Extensiones recomendadas
1. Modelado de ruido: implementar canales de dephasing/depolarizing y ver impacto en estimación de r.
2. Qiskit integration: transpilar circuitos y ejecutar en simuladores con ruido o hardware remoto.
3. Métricas: tasa de éxito vs shots, t, tipo de a, y vs nivel de ruido.
4. Visualizaciones: mapa de calor de picos de fase, fidelity estimators.

## Archivos generados (ejemplo)
- summary_N15.json
- result_N15_a13.json
- hist_N15_a13.png

## Cómo interpretar resultados
- Si el pipeline encontró 'factors' en alguno de los intentos: éxito pedagógico.
- Si no: analizar top-measured outcomes, probar mayor 't' o más 'shots', o revisar elección de 'a'.

