# STT SIGNALS - IV_CONV / IV ATM / PUT SKEW / SQI_V2

Dashboard de 3 senales de regimen de mercado validadas contra STT V9 (PUT BWB
+K1 -2K2 +K3, DTE 150-170, **1.633 trades / 1.226 dias unicos**, 2019-2025).

**Live:** https://manumartinb.github.io/STT_SIGNALS/

## Las 3 senales

Todas son senales de **mercado** medidas a `dte_target=160` sobre el snapshot de las
10:30 ET, con el RAW suavizado por **mediana movil trailing de 3 dias** (ex-ante) antes
de transformarse a la escala 0-100.

| Senal | Definicion | Transformacion | r vs PnL_d030 |
|---|---|---|---|
| IV_CONV | `(iv_5d + iv_30d)/2 - iv_15d` (concavidad de la PUT smile, Werner) | percentil **expanding** | +0.20 |
| IV ATM | nivel IV ATM `iv_50d` (eje vol, migrado desde VIX en 2026-05) | percentil **expanding** | +0.46 |
| PUT SKEW NIVEL | `skew_25d_vs50` (prima de cola en puts, 25d vs ATM) | **referencia FIJA** (searchsorted) | +0.36 |

Bandas: FAVORABLE >=80, NEUTRAL 20-80, ADVERSO <=20.
Umbrales de PUT SKEW en unidades crudas: ADVERSO <= 0.03558, FAVORABLE >= 0.05780.

## Decisiones de diseno (2026-09-02)

Las tres salen de una auditoria de la pagina + un @APR de ventanas. Ninguna es heredada.

1. **Una variable, una definicion.** El panel de arriba y las tablas de abajo usan
   exactamente las mismas tres senales. Antes las tablas puntuaban IV_CONV con las IV de
   las patas del trade (`(iv_k1+iv_k3)/2 - iv_k2`) y el panel usaba el proxy de mercado:
   coincidian solo el 78% de los dias en el umbral P80, asi que la tabla prometia un
   resultado que la luz no seleccionaba igual (edge desplegable +1.40 frente al +1.83
   publicado). Coste asumido: `r(IV_CONV)` baja de +0.32 a +0.20.
   La version **trade-specific no se ha perdido**: vive donde si existe un trade concreto
   -- la madre `STT_CLASSIC_V9_MERGED_T0_mediana.csv` (columna `IV_CONVEXITY`) y el scanner
   LIVE, donde alimenta el score `SQI_V2` (gate operativo `SQI_V2_PCTLE >= 80`).
2. **PUT SKEW contra referencia FIJA, no percentil expanding.** El @APR mostro que el
   valor crudo BATE al percentil expanding (+0.0631 de Spearman, CI95 [+0.0418, +0.0864]
   por bootstrap por dia) y que el expanding RESTA informacion (partial(exp|raw) = -0.128;
   k-fold 5/5 frente a 4/5). El `searchsorted` contra una distribucion congelada
   (`PS_SKEW_REF_FROZEN.npz`, n=1929) es la version monotona del crudo en escala 0-100:
   conserva su poder intacto y mantiene la lectura 0-100. `r` sube de +0.25 a +0.36.
3. **Sin ventanas rolling.** Se evaluaron 504 / 252 / 126 dias frente al expanding: las
   12 comparaciones pierden y 11 son significativas por bootstrap por dia. Acortar la
   ventana no redistribuye el edge, lo destruye.

## Como leer la evidencia

- Debajo de **cada cohorte** va su **reparto por anio**, en ambar si un solo anio pesa
  mas del 60%. Varias cohortes "de oro" son en la practica un unico regimen (2020).
- Cada **tarjeta** indica cuando marco FAVORABLE por ultima vez. IV_CONV e IV ATM se
  miden contra toda la historia desde 2019: el liston lo fijaron 2020 y 2022, asi que una
  fecha lejana no significa que la senal este rota, sino que no estamos cerca de un
  extremo historico.
- `PF = inf` significa "sin trades perdedores en la cohorte", no un numero grande.
- **IV_CONV se invierte en 2019** (r -0.52 ese ano). De ahi salen sus cohortes BOT
  positivas: es un ano en que la senal falla, no una senal de doble filo.

## Hallazgos vigentes

- **El par mas ortogonal es IVC x IV ATM** (rho ~+0.29). IV ATM x PUT SKEW comparten
  mucho (rho ~+0.73): combinarlas diversifica poco.
- **IV ATM es el predictor mas fuerte en solitario** (r ~+0.46); PUT SKEW el segundo
  (r ~+0.36). Migracion VIX -> IV ATM: r +0.45 vs +0.39, tenor casado al trade.
- **Triple AND** (las 3 >=P80): N=37 trades y 25 de 2020. Detector de crisis, no setup
  recurrente. El "HIERRO" no tiene muestra para ser una regla de no-entrada.
- **SDEX descartado**: no transfiere a STT (r=-0.11, patron en U).
- **BB sobre IV_CONV descartado**: inferior al percentil expanding.
- **Alerta 2025**: en el @APR el peor mes es 2025-02 para los 20 arms evaluados y el
  ultimo fold del k-fold sale negativo en 18 de 20. No es un problema de ventana.

## Estructura

- **Seccion A - SOLAS**: cortes percentil (TOP/BOT) de cada senal vs RAW.
- **Seccion B - 2 a 2**: joint terciles 3x3 + composite AND/OR.
- **Seccion C - 3 a 3**: triple AND, >=2 de 3, OR, hierro triple.
- **Seccion D - correlaciones**: matriz 3x3 + r vs PnL.

Todas las tablas y charts en doble panel **mean + median**.

## Archivos

- `index.html` - dashboard (lee data.json)
- `data.json` - datos serializados
- `evidence/` - PNGs (trayectorias, heatmaps, composites)
- `update_dashboard.py` - regenera data.json + evidencia (tablas + panel)
- `daily_refresh.py` - worker diario: refresca SOLO `series` + `latest` y hace push
- `stt_heal.py` - saneamiento de dias con glitch de tasa r=0 (no-op si no hay)
- `PS_SKEW_REF_FROZEN.npz` - referencia congelada de PUT SKEW (no regenerar a la ligera:
  cambiarla mueve todas las bandas historicas)

## Fuentes

- `Backtests DATABASE/STT/STT_CLASSIC_V9_MERGED_T0_mediana.csv` (PnL_d001..d030)
- `Skew/SKEW_PUT_ENRICHED.csv` con `dte_target=160` (`iv_5d`, `iv_15d`, `iv_25d`,
  `iv_30d`, `iv_50d`, `skew_25d_vs50`)
- `FINAL DATA/SP_SPX_CLOSE_HISTORICAL_PRICES.csv` (eje secundario del grafico)

## Documentacion relacionada

- `ESTRATEGIAS/SQI_V2_Formula_(STT).md` - el score de calidad trade-level
- Auditoria y @APR: `memory/auditoria_360_sistema_manumb_stt-signals_20260902.md` y
  `memory/analisis_predictabilidad_robustez_ventanas_stt_20260902.md`
