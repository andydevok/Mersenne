# ========================================================================
# SCRIPT: CAZA A CIEGAS AUTÓNOMA + FILTRO FINO v54.0
#
# Características:
# 1. Script auto-contenido (con los ceros de Riemann hardcodeados)
#    para facilitar su distribución.
# 2. Realizar una "Caza a Ciegas" para redescubrir M38 en un territorio de
#    1,000,000 de ancho usando el método de "Mínima Desviación 2D".
# 3. Aplicar un "Filtro Fino" (Proximidad a 2^k, Aislamiento, Sesgo Modular)
#    a los 10 mejores candidatos encontrados para un análisis de robustez.
# 4. Presentar un veredicto final híbrido.
#
# Autor: Andrés Sebastián Pirolo
# Fecha: 03 de octubre de 2025
# ========================================================================

# --- INSTALACIÓN DE LIBRERÍAS ---
!pip install mpmath sympy pandas

import numpy as np
import pandas as pd
import time
import mpmath
from sympy import primerange
from tqdm import tqdm

print("--- INICIANDO SCRIPT: Caza a Ciegas Autónoma v54.0 ---")
script_start_time = time.time()

try:
    # =======================================================
    # BLOQUE 1: CONFIGURACIÓN Y DATOS HARDCODEADOS
    # =======================================================
    CONFIG = {
        'PRECISION_DPS': 100,
        'TARGET_P_REAL': 6972593, # M38
        'HUNTING_GROUND': (6500000, 7500000),
        'BROAD_SCAN_SAMPLE_SIZE': 50000,
        'SNIPER_SHOT_RADIUS': 2000,
        'TOP_N_CANDIDATES': 10,
        'ISOLATION_RADIUS': 5000
    }
    mpmath.mp.dps = CONFIG['PRECISION_DPS']
    print(f"Precisión de mpmath: {CONFIG['PRECISION_DPS']} dígitos.")
    
    # --- Ceros de Riemann Hardcodeados ---
    RIEMANN_ZEROS = [
        14.134725142, 21.022039639, 25.010857580, 30.424876126, 32.935061588,
        37.586178159, 40.918719012, 43.327073281, 48.005150881, 49.773832478,
        52.970321478, 56.446247697, 59.347044003, 60.831778525, 65.112544048,
        67.079810529, 69.546401711, 72.067157674, 75.704690699, 77.144840069,
        79.337375020, 82.910380854, 84.735492981, 87.425274613, 88.809111208,
        92.491899271, 94.651344041, 95.870634228, 98.831194218, 101.317851006,
        103.725538040, 105.446623052, 107.168611184, 111.029535543, 111.874659177,
        114.320220915, 116.226680321, 118.790782866, 121.370125002, 122.946829294,
        124.256818554, 127.516683880, 129.578704200, 131.087688531, 133.497737203,
        134.756509753, 138.116042055, 139.7362088952, 141.123707404, 143.111845808,
        146.000982487, 147.422765343, 150.053520421, 150.925257612, 153.024693811,
        156.112909294, 157.597591818, 158.849988171, 161.188964138, 163.030709687,
        165.537069188, 167.184439978, 169.094515416, 169.911976479, 173.411536520,
        174.754191523, 176.441434298, 178.377407776, 179.916484020, 182.207078484,
        184.874467848, 185.598783678, 187.228922584, 189.416158656, 192.026656361,
        193.079726604, 195.265396680, 196.876481841, 198.015309676, 201.264751944,
        202.493594514, 204.189671803, 205.394697202, 207.906258888, 209.576509717,
        211.690862595, 213.347919360, 214.547044783, 216.169538508, 219.067596349,
        220.714918839, 221.430705555, 224.007000255, 224.983324670, 227.421444280,
        229.337413306, 231.250188700, 231.987235253, 233.693404179, 236.524229666
    ]
    zeros_lens_mp = [mpmath.mpf(z) for z in RIEMANN_ZEROS]
    print(f"✔ {len(zeros_lens_mp)} Ceros de Riemann cargados en memoria.")

    # ===============================================
    # BLOQUE 3: FUNCIONES
    # ===============================================
    
    # --- Funciones Geométricas ---
    def get_2d_jump_distance(exponents, desc_text=""):
        # (código idéntico a v51.0)
        distances_2d = []
        num_zeros_mp = mpmath.mpf(len(zeros_lens_mp))
        for p in tqdm(exponents, desc=desc_text):
            p_mp = mpmath.mpf(int(p))
            log_gen = p_mp * mpmath.log(2)
            log_mp = log_gen + mpmath.log(1 - mpmath.power(2, -p_mp))
            sc_gen, ss_gen, sc_mp, ss_mp = mpmath.mpf(0), mpmath.mpf(0), mpmath.mpf(0), mpmath.mpf(0)
            for z in zeros_lens_mp:
                sc_gen += mpmath.cos(z * log_gen); ss_gen += mpmath.sin(z * log_gen)
                sc_mp += mpmath.cos(z * log_mp); ss_mp += mpmath.sin(z * log_mp)
            dx = (sc_mp - sc_gen) / num_zeros_mp
            dy = (ss_mp - ss_gen) / num_zeros_mp
            distances_2d.append(mpmath.sqrt(dx**2 + dy**2))
        return distances_2d

    # --- Funciones para Filtro Fino ---
    def calculate_proximity_to_2k(p):
        powers_of_2 = np.array([2**k for k in range(20, 30)])
        return np.min(np.abs(p - powers_of_2))

    def calculate_primal_isolation(p, radius):
        return len(list(primerange(p - radius, p + radius)))

    def check_modular_bias(p):
        return 1 if (p % 4 == 1) else 0

    # ===============================================
    # BLOQUE 4: CALIBRACIÓN Y CAZA A CIEGAS
    # ===============================================
    print("\n--- FASE DE CALIBRACIÓN: Aprendiendo la 'Ley de Decaimiento 2D' ---")
    CALIBRATION_EXPONENTS = np.array([13, 17, 19, 31, 61, 89, 107, 127])
    calibration_distances_2d = get_2d_jump_distance(CALIBRATION_EXPONENTS, desc_text="Calibrando Ley 2D")
    log_p_calib = np.log(CALIBRATION_EXPONENTS.astype(float))
    log_dist_2d_calib = [float(mpmath.log(d)) if d > 0 else -999 for d in calibration_distances_2d]
    m, c = np.polyfit(log_p_calib, log_dist_2d_calib, 1)
    print(f"✔ 'Ley de Decaimiento 2D' aprendida: log(Dist_2D) ≈ {m:.4f} * log(p) + {c:.4f}")

    def get_deviation_score(p_candidate, dist_2d_candidate_mp, m, c):
        if dist_2d_candidate_mp < 1e-99: return -999.0
        log_p = mpmath.log(p_candidate)
        log_dist_real = mpmath.log(dist_2d_candidate_mp)
        log_dist_esperada = m * log_p + c
        return float(log_dist_real - log_dist_esperada)

    print(f"\n--- INICIANDO CAZA A CIEGAS PARA M38 ---")
    HUNTING_GROUND = CONFIG['HUNTING_GROUND']
    all_candidates = np.array(list(primerange(HUNTING_GROUND[0], HUNTING_GROUND[1])))
    
    # --- FASE 1: RECONOCIMIENTO ---
    np.random.seed(42)
    sample_size = min(CONFIG['BROAD_SCAN_SAMPLE_SIZE'], len(all_candidates))
    sample_indices = np.random.choice(len(all_candidates), sample_size, replace=False)
    sample_candidates = np.sort(all_candidates[sample_indices])
    sample_distances_mp = get_2d_jump_distance(sample_candidates, desc_text="Fase 1 (Muestra a ciegas)")
    deviation_scores = [get_deviation_score(p, d, m, c) for p, d in zip(sample_candidates, sample_distances_mp)]
    preliminary_winner = sample_candidates[np.argmin(deviation_scores)]
    
    # --- FASE 2: PRECISIÓN ---
    sniper_start = preliminary_winner - CONFIG['SNIPER_SHOT_RADIUS']
    sniper_end = preliminary_winner + CONFIG['SNIPER_SHOT_RADIUS']
    elite_candidates_mask = (all_candidates >= sniper_start) & (all_candidates <= sniper_end)
    elite_candidates = all_candidates[elite_candidates_mask]
    elite_distances_mp = get_2d_jump_distance(elite_candidates, desc_text="Fase 2 (Élite)")
    elite_deviation_scores = [get_deviation_score(p, d, m, c) for p, d in zip(elite_candidates, elite_distances_mp)]
    
    df_hunt_results = pd.DataFrame({'p': elite_candidates, 'score': elite_deviation_scores})
    df_hunt_results.sort_values('score', inplace=True)
    top_candidates = df_hunt_results.head(CONFIG['TOP_N_CANDIDATES'])
    
    print(f"✔ Caza a ciegas completada. Top {CONFIG['TOP_N_CANDIDATES']} candidatos identificados.")

    # ===============================================
    # BLOQUE 5: APLICACIÓN DEL FILTRO FINO
    # ===============================================
    print("\n--- APLICANDO FILTRO FINO A LOS MEJORES CANDIDATOS ---")
    final_analysis = []
    for p in tqdm(top_candidates['p'], desc="Aplicando Filtros Clásicos"):
        analysis = {
            'Candidato (p)': p,
            'Score_Geométrico': top_candidates[top_candidates['p'] == p]['score'].iloc[0],
            'Filtro_Proximidad_2k': calculate_proximity_to_2k(p),
            'Filtro_Aislamiento': calculate_primal_isolation(p, CONFIG['ISOLATION_RADIUS']),
            'Filtro_Modular': check_modular_bias(p)
        }
        final_analysis.append(analysis)
    
    df_final = pd.DataFrame(final_analysis)
    
    # Calcular Ranks
    df_final['Rank_Geo'] = df_final['Score_Geométrico'].rank(method='min', ascending=True)
    df_final['Rank_2k'] = df_final['Filtro_Proximidad_2k'].rank(method='min', ascending=True)
    df_final['Rank_Aislamiento'] = df_final['Filtro_Aislamiento'].rank(method='min', ascending=True)
    df_final['Rank_Modular'] = df_final['Filtro_Modular'].rank(method='min', ascending=False)
    df_final['Rank_Total'] = df_final[['Rank_Geo', 'Rank_2k', 'Rank_Aislamiento', 'Rank_Modular']].sum(axis=1)
    df_final.sort_values('Rank_Total', inplace=True)

    # =================================
    # BLOQUE 6: REPORTE FINAL
    # =================================
    print("\n\n" + "="*80)
    print("--- REPORTE FINAL: CAZA A CIEGAS + FILTRO FINO (v54.0) ---")
    print("="*80)

    TARGET_P = CONFIG['TARGET_P_REAL']
    final_winner_p = int(df_final.iloc[0]['Candidato (p)'])
    error_abs = abs(TARGET_P - final_winner_p)
    error_rel = (error_abs / TARGET_P) * 100

    print(f"\nTerritorio de Caza: {HUNTING_GROUND[0]:,} a {HUNTING_GROUND[1]:,}")
    print(f"Objetivo Real (M38):  {TARGET_P:,}")
    print(f"Mejor Candidato Geométrico (pre-filtro): {top_candidates['p'].iloc[0]:,}")
    print(f"Mejor Candidato Híbrido (post-filtro):   {final_winner_p:,}")
    
    print("\n--- TABLA DE ANÁLISIS HÍBRIDO (MEJORES CANDIDATOS) ---")
    pd.set_option('display.float_format', '{:,.2f}'.format)
    print(df_final.to_string(index=False))

    print("\n--- VEREDICTO FINAL ---")
    if TARGET_P == final_winner_p:
        print("✅ ¡ÉXITO TOTAL! El filtro híbrido ha redescubierto M38 con perfecta precisión.")
    elif error_rel < 0.1:
        print(f"✅ ¡ÉXITO PARCIAL! El modelo híbrido encontró un candidato con un error relativo de solo {error_rel:.4f}%.")
    else:
        print(f"❌ FALLO. El modelo híbrido no logró localizar el objetivo. El error fue de {error_rel:.4f}%.")
        
    print("="*80)

except Exception as e:
    print(f"\n❌ Ocurrió un error inesperado durante la ejecución: {e}")
    import traceback
    traceback.print_exc()

finally:
    script_end_time = time.time()
    total_time = script_start_time - script_end_time
    print(f"\n\n--- SCRIPT FINALIZADO (Tiempo total: {abs(total_time):.2f}s) ---")
