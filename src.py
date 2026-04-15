import numpy as np
import pandas as pd
from fpdf import FPDF
import os

COPRIFERRO = 0.05   # [m]


def calcola_area_armatura(df_armature: pd.DataFrame) -> float:
    df_clean = df_armature.dropna()
    if df_clean.empty:
        return 0.0
    aree = df_clean['Num'] * (np.pi * df_clean['Diam_mm'] ** 2 / 4)
    return round(aree.sum() / 100, 2)


def calc_ur(T: float, A: float, f_yd: float) -> float:
    if T <= 0:
        return 0.0
    if A <= 0:
        return float('inf')
    return (T * 10.0 / A) / f_yd   # T[kN], A[cm²], f_yd[MPa]


def calc_ur_c(C: float, b: float, l_s: float, fcd_eff: float) -> float:
    """UR puntone: σ_c = C/(b·l_s) ≤ ν'·fcd
    C[kN], b[m], l_s[m] larghezza nodo, fcd_eff[MPa]"""
    if C <= 0:
        return 0.0
    if b <= 0 or l_s <= 0 or fcd_eff <= 0:
        return float('inf')
    return C / (b * l_s * fcd_eff * 1000.0)


def analizza_modello_combinato(
    geo: dict, carichi: dict, mat: dict,
    A_T1: float, A_T2a: float, A_T1_prime: float,
    alpha_deg: float, A_T3: float, k_A: float
) -> dict:
    """
    Verifica sella Gerber – modelli paralleli A e B (Angotti Cap. 10,
    Prospetti 10.15 e 10.16).

    Sistema di riferimento disegno
    ───────────────────────────────
    x→ destra, y = 0 bordo superiore trave, y negativo verso il basso.
    Dente : x ∈ [0, a],     y ∈ [-h1, 0]
    Trave : x ∈ [a, L_tot], y ∈ [-H,  0]

    Modello A – staffe di sospensione (Prospetto 10.15)
    ────────────────────────────────────────────────────
    Nodi:
      N1  = (a-av, -(h1-c))   appoggio (bearing plate)
      N3  = (a,    -c)         spigolo superiore dente
      N4  = (a,    -(H-c))     nodo inferiore trave alla sezione di cambio
      N2A = (a+z2, -(H-c)+z2) nodo interno (puntone C4, 45°)  ← nel disegno

    Forze:
      θ₁ = arctan(z₁/av)          angolo di C1 dalla orizzontale
      C1  = Ra / sin(θ₁)          puntone dente
      T1  = Ra·av/z₁ + H_Ed       tirante inferiore dente  ← solo Mod. A + H_Ed
                                   (Mod. B non contribuisce: C'1 è verticale)
      T2a = Ra                     staffe verticali nel dente
      T2A = Ra·(1 + av/(2z₁))     tirante inferiore trave, contributo Mod. A
             (Angotti: con θ₂=45°)

    Modello B – ferri piegati (Prospetto 10.16)
    ─────────────────────────────────────────────
    Nodi:
      N1' = N1 (stesso appoggio)
      N2B = (a-av, -c)             testa di C'1 (verticale)
      N3B = (a-av + z₂/tan α, -(H-c))   ancoraggio ferri piegati (fondo trave)
      N4B = (a-av + 2z₂/tan α, -c)      testa di C'3

    Forze:
      C'1 = Rb                     puntone verticale nel dente
      T'1 = Rb / sin(α)            ferri piegati (tirante inclinato verso basso-destra)
      T2B = 2·Rb / tan(α)          tirante inferiore trave, contributo Mod. B
             (Angotti: Prospetto 10.16 generalizzato da 45°)

    Tiranti combinati
    ─────────────────
      T1 = Ra·av/z₁ + H_Ed             tirante inferiore dente
      T3 = Ra·(1+av/2z₁) + 2Rb/tan α + H_Ed   tirante inferiore trave
    """
    c  = COPRIFERRO
    h1 = geo['h1']
    H  = geo['H']
    a  = geo['a']
    av = geo['av']
    b  = geo['b']
    l_b = geo.get('l_b', 0.20)      # larghezza piastra di appoggio [m]
    z1 = h1 - 2 * c                 # braccio interno dente
    z2 = H  - 2 * c                 # braccio interno trave principale

    Ved = carichi['Ved']
    Hed = carichi['Hed']

    alpha_rad = np.radians(alpha_deg)
    sin_a = np.sin(alpha_rad)
    cos_a = np.cos(alpha_rad)
    tan_a = np.tan(alpha_rad)

    f_yd    = mat['fyk'] / mat['gamma_s']
    fcd     = mat['fck'] / mat['gamma_c']               # [MPa]
    nu_p    = 0.6 * (1.0 - mat['fck'] / 250.0)         # fattore eff. EC2 §6.5.2
    fcd_eff = nu_p * fcd                                # resist. eff. puntoni [MPa]

    Ra = k_A * Ved
    Rb = (1.0 - k_A) * Ved

    # ── MODELLO A ──────────────────────────────────────────────────────────────
    theta1 = np.arctan(z1 / av) if av > 0 else np.pi / 2

    C1_A = Ra / np.sin(theta1) if Ra > 0 else 0.0
    T1_A = Ra * av / z1       if Ra > 0 and z1 > 0 else 0.0   # = Ra/tan(θ₁)
    T2a  = Ra
    T2_A = Ra * (1.0 + av / (2.0 * z1)) if Ra > 0 and z1 > 0 else 0.0

    # Puntoni aggiuntivi Mod. A
    C2_A = C1_A                                           # simmetrico: θ2 = θ1
    dH   = H - h1                                         # dislivello dente→trave
    L_C3 = np.sqrt(av ** 2 + dH ** 2) if dH > 0 else 0.0
    C3_A = Ra * L_C3 / dH if (Ra > 0 and dH > 0) else 0.0
    C4_A = Ra * np.sqrt(2.0) if Ra > 0 else 0.0          # 45° nel corpo trave

    # ── MODELLO B ──────────────────────────────────────────────────────────────
    T1_prime = Rb / sin_a        if Rb > 0 else 0.0
    T2_B     = 2.0 * Rb / tan_a if Rb > 0 else 0.0

    # Puntoni aggiuntivi Mod. B
    # C'1 = Rb (puntone verticale, già nel dict come 'Rb')
    C2_B = Rb / tan_a if Rb > 0 else 0.0   # compressione orizzontale sup.
    C3_B = Rb / sin_a if Rb > 0 else 0.0   # = T1_prime (puntone diagonale)

    # ── VERIFICHE PUNTONI (σ_c = C/(b·l_s) ≤ ν'·fcd) ─────────────────────────
    # Larghezze nodo (perpendicolari al puntone):
    #   C1, C2  : nodo sull'appoggio  → l_s = l_b·sin(θ1) + 2c·cos(θ1)
    #   C3, C4  : approssimazione con l_b
    #   C'1     : verticale sull'appoggio → l_s = l_b
    #   C'2     : orizz. in sommità → l_s = 2c  (zona nodo superiore)
    #   C'3     : nodo inferiore → l_s = l_b·sin(α) + 2c·cos(α)
    sin_t1 = np.sin(theta1);  cos_t1 = np.cos(theta1)
    ls_C1  = l_b * sin_t1 + 2 * c * cos_t1   # [m] larghezza nodo C1/C2
    ls_C3  = l_b                               # [m] approx.
    ls_C4  = l_b                               # [m] approx.
    ls_C1B = l_b                               # C'1 verticale
    ls_C2B = max(2 * c, 0.05)                 # C'2 orizz. sommità
    ls_C3B = l_b * sin_a + 2 * c * cos_a     # C'3 diagonale

    ur_C1_A = calc_ur_c(C1_A,  b, ls_C1,  fcd_eff)
    ur_C2_A = calc_ur_c(C2_A,  b, ls_C1,  fcd_eff)   # stessa geometria nodo
    ur_C3_A = calc_ur_c(C3_A,  b, ls_C3,  fcd_eff)
    ur_C4_A = calc_ur_c(C4_A,  b, ls_C4,  fcd_eff)
    ur_C1_B = calc_ur_c(Rb,    b, ls_C1B, fcd_eff)
    ur_C2_B = calc_ur_c(C2_B,  b, ls_C2B, fcd_eff)
    ur_C3_B = calc_ur_c(C3_B,  b, ls_C3B, fcd_eff)

    # ── TIRANTI TOTALI ─────────────────────────────────────────────────────────
    # Mod. B non contribuisce al tirante del dente (C'1 è verticale → nessuna
    # componente orizzontale all'appoggio).
    T1 = T1_A + Hed
    T3 = T2_A + T2_B + Hed

    # ── VERIFICHE ──────────────────────────────────────────────────────────────
    ur_T1       = calc_ur(T1,       A_T1,       f_yd)
    ur_T2a      = calc_ur(T2a,      A_T2a,      f_yd)
    ur_T1_prime = calc_ur(T1_prime, A_T1_prime, f_yd)
    ur_T3       = calc_ur(T3,       A_T3,       f_yd)

    all_urs = [ur_T1, ur_T2a, ur_T1_prime, ur_T3,
               ur_C1_A, ur_C2_A, ur_C3_A, ur_C4_A,
               ur_C1_B, ur_C2_B, ur_C3_B]
    finite_urs = [u for u in all_urs if u != float('inf')]
    esito = all(u <= 1.0 for u in finite_urs)

    # ── NODI PER IL DISEGNO ────────────────────────────────────────────────────
    N1   = (a - av,                    -(h1 - c))   # appoggio (bearing plate)
    # Modello A
    N3   = (a,                          -c)          # spigolo superiore dente
    N4   = (a,                          -(H - c))    # nodo inferiore trave (sez. step)
    N2A  = (a + z2,                     -c)          # testa puntone C4 (45°)
    # Modello B
    N2B  = (a - av,                     -c)          # testa puntone verticale C'1
    x3B  = (a - av) + z2 / tan_a if Rb > 0 else a + 0.5
    N3B  = (x3B,                        -(H - c))    # ancoraggio ferri piegati (basso)
    x4B  = (a - av) + 2.0 * z2 / tan_a if Rb > 0 else a + 1.0
    N4B  = (x4B,                        -c)          # testa C'3 (alto destra)

    return {
        "Ra": round(Ra, 2), "Rb": round(Rb, 2), "k_A": k_A,
        # Puntoni Mod. A (forze)
        "C1_A":  round(C1_A, 2),
        "C2_A":  round(C2_A, 2),
        "C3_A":  round(C3_A, 2),
        "C4_A":  round(C4_A, 2),
        # Puntoni Mod. B (forze) — C'1 = Rb
        "C2_B":  round(C2_B, 2),
        "C3_B":  round(C3_B, 2),
        # UR puntoni Mod. A
        "ur_C1_A": round(ur_C1_A, 3),
        "ur_C2_A": round(ur_C2_A, 3),
        "ur_C3_A": round(ur_C3_A, 3),
        "ur_C4_A": round(ur_C4_A, 3),
        # UR puntoni Mod. B
        "ur_C1_B": round(ur_C1_B, 3),
        "ur_C2_B": round(ur_C2_B, 3),
        "ur_C3_B": round(ur_C3_B, 3),
        # Larghezze nodo usate
        "ls_C1":  round(ls_C1,  4),
        "ls_C3B": round(ls_C3B, 4),
        # Materiali calcestruzzo
        "fcd_eff": round(fcd_eff, 2),
        "nu_p":    round(nu_p, 4),
        "l_b":     l_b,
        # Tiranti intermedi
        "T1_A":      round(T1_A, 2),
        "T2a":       round(T2a, 2),
        "T2_A":      round(T2_A, 2),
        "T1_prime":  round(T1_prime, 2),
        "T2_B":      round(T2_B, 2),
        "T1":        round(T1, 2),
        "T3":        round(T3, 2),
        "ur_T1":       round(ur_T1, 3),
        "ur_T2a":      round(ur_T2a, 3),
        "ur_T1_prime": round(ur_T1_prime, 3),
        "ur_T3":       round(ur_T3, 3),
        "theta1_deg":  round(np.degrees(theta1), 1),
        "alpha_deg":   alpha_deg,
        "z1":          round(z1, 3),
        "z2":          round(z2, 3),
        "f_yd":        round(f_yd, 2),
        "esito":       esito,
        # nodi
        "N1": N1, "N3": N3, "N4": N4, "N2A": N2A,
        "N2B": N2B, "N3B": N3B, "N4B": N4B,
    }


def genera_pdf_report(ris: dict, img_path: str = None) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 15)
    pdf.cell(0, 10, "Relazione Sella Gerber - Modelli S&T (Angotti Cap. 10)",
             ln=True, align='C')
    pdf.ln(3)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 7,
             f"Ripartizione: {ris['k_A']*100:.0f}% Mod.A (Staffe)  +  "
             f"{(1-ris['k_A'])*100:.0f}% Mod.B (Ferri piegati {ris['alpha_deg']:.0f} gradi)",
             ln=True)
    pdf.ln(2)
    rows = [
        ("theta1",     f"{ris['theta1_deg']:.1f} gradi"),
        ("C1_A",       f"{ris['C1_A']:.1f} kN  (puntone dente)"),
        ("T1 dente",   f"{ris['T1']:.1f} kN  - UR {ris['ur_T1']*100:.1f}%"),
        ("T2a staffe", f"{ris['T2a']:.1f} kN  - UR {ris['ur_T2a']*100:.1f}%"),
        ("T1' ferri",  f"{ris['T1_prime']:.1f} kN  - UR {ris['ur_T1_prime']*100:.1f}%"),
        ("T3 trave",   f"{ris['T3']:.1f} kN  - UR {ris['ur_T3']*100:.1f}%"),
    ]
    pdf.set_font("Arial", '', 10)
    for label, val in rows:
        pdf.cell(50, 6, f"  {label}:", border=0)
        pdf.cell(0, 6, val, ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", 'B', 13)
    esito_txt = "VERIFICHE SODDISFATTE" if ris['esito'] else "VERIFICHE NON SODDISFATTE"
    pdf.set_text_color(0, 140, 0) if ris['esito'] else pdf.set_text_color(200, 0, 0)
    pdf.cell(0, 9, f"ESITO: {esito_txt}", ln=True, align='C')
    pdf.set_text_color(0, 0, 0)
    if img_path and os.path.exists(img_path):
        pdf.ln(6)
        pdf.image(img_path, x=12, w=180)
    return pdf.output(dest='S').encode('latin-1')
