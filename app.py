"""
Verifica Sella Gerber – Modelli S&T paralleli (Angotti Cap. 10, fig. 10.72)
============================================================================
Modello A : puntone inclinato C1 + staffe verticali T2a   (fig. 10.72a)
Modello B : puntone verticale C'1 + ferri piegati T'1     (fig. 10.72b)
Combinato : Ra + Rb = V_Ed  (ripartizione libera tramite slider)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tempfile, os

from src import calcola_area_armatura, analizza_modello_combinato, genera_pdf_report

st.set_page_config(page_title="Sella Gerber S&T", layout="wide")
st.title("Verifica Sella Gerber – Modelli Tirante-Puntone (Angotti Cap. 10)")
st.caption(
    "Mod. **A**: puntone C1 inclinato + staffe T2a  |  "
    "Mod. **B**: puntone verticale C'1 + ferri piegati T'1  |  "
    "Rif.: Angotti Prospetti 10.15–10.16, fig. 10.72"
)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parametri di Input")
    with st.expander("1. Geometria [m]", expanded=True):
        H  = st.number_input("H – Altezza trave principale", value=1.00, step=0.05, min_value=0.20)
        h1 = st.number_input("h1 – Altezza dente sella",    value=0.50, step=0.05, min_value=0.10)
        b  = st.number_input("b – Larghezza sezione",        value=0.40, step=0.05, min_value=0.10)
        a  = st.number_input("a – Lunghezza dente",          value=0.40, step=0.05, min_value=0.05)
        av = st.number_input("av – Dist. carico da spigolo", value=0.20, step=0.01, min_value=0.01)
        l_b = st.number_input("l_b – Lungh. piastra appoggio", value=0.20, step=0.01, min_value=0.05)
    with st.expander("2. Azioni [kN]", expanded=True):
        Ved = st.number_input("V_Ed – Reazione appoggio", value=350.0, step=10.0, min_value=0.0)
        Hed = st.number_input("H_Ed – Forza orizzontale", value=50.0,  step=5.0,  min_value=0.0)
    with st.expander("3. Materiali", expanded=True):
        fck = st.number_input("f_ck [MPa]", value=30.0,  step=1.0,  min_value=12.0)
        fyk = st.number_input("f_yk [MPa]", value=450.0, step=10.0, min_value=200.0)
    c_cov = 0.05
    z1_val = h1 - 2 * c_cov
    z2_val = H  - 2 * c_cov
    st.info(f"z₁ (dente) = {z1_val:.3f} m\nz₂ (trave) = {z2_val:.3f} m")
    if av >= z1_val:
        st.warning("av ≥ z1: verificare geometria (angolo θ1 molto piccolo).")

geo     = {'H': H, 'h1': h1, 'b': b, 'a': a, 'av': av, 'l_b': l_b}
carichi = {'Ved': Ved, 'Hed': Hed}
mat     = {'fck': fck, 'fyk': fyk, 'gamma_c': 1.5, 'gamma_s': 1.15}

# ─── ARMATURE ─────────────────────────────────────────────────────────────────
st.subheader("Configurazione Armature")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("**T1 – Tirante inferiore dente**")
    st.caption("Armatura orizzontale al piede del dente")
    df_T1 = st.data_editor(pd.DataFrame([{"Num": 4, "Diam_mm": 16}]),
                           num_rows="dynamic", hide_index=True, key="t1")
    A_T1 = calcola_area_armatura(df_T1)
    st.metric("Area T1", f"{A_T1} cm²")

with c2:
    st.markdown("**T2a – Staffe sospensione (Mod. A)**")
    st.caption("Armatura verticale nel dente")
    alpha = st.number_input("Inclinazione ferri piegati α [°]",
                            min_value=20.0, max_value=80.0, value=45.0, step=5.0)
    df_T2a = st.data_editor(pd.DataFrame([{"Num": 4, "Diam_mm": 12}]),
                            num_rows="dynamic", hide_index=True, key="t2a")
    A_T2a = calcola_area_armatura(df_T2a)
    st.metric("Area T2a", f"{A_T2a} cm²")

with c3:
    st.markdown("**T'1 – Ferri piegati (Mod. B)**")
    st.caption(f"Armatura inclinata a {alpha:.0f}°")
    df_T1p = st.data_editor(pd.DataFrame([{"Num": 2, "Diam_mm": 16}]),
                            num_rows="dynamic", hide_index=True, key="t1p")
    A_T1_prime = calcola_area_armatura(df_T1p)
    st.metric("Area T'1", f"{A_T1_prime} cm²")

with c4:
    st.markdown("**T3 – Tirante trave principale**")
    st.caption("Armatura inferiore trave vicino appoggio")
    df_T3 = st.data_editor(pd.DataFrame([{"Num": 4, "Diam_mm": 20}]),
                           num_rows="dynamic", hide_index=True, key="t3")
    A_T3 = calcola_area_armatura(df_T3)
    st.metric("Area T3", f"{A_T3} cm²")

# ─── DISEGNO TRALICCIO ────────────────────────────────────────────────────────
def disegna_traliccio(ris: dict, geo: dict, carichi: dict, k_A: float) -> go.Figure:
    """
    Riproduce fedelmente la fig. 10.72 di Angotti.

    Modello A (fig. a)
    ──────────────────
    N1  = appoggio (bearing plate)
    N3  = spigolo superiore dente  ← testa di C1
    N4  = nodo inferiore alla sezione di step (x=a, fondo trave)
    N2A = nodo interno trave (C4 a 45° da N4)
    Aste: C1 (N1→N3, inclinato), T2a (verticale nel dente),
          T1 (breve orizzontale, ancoraggio dente),
          C4 (N4→N2A, puntone 45° in trave), T3 (orizzontale fondo trave)

    Modello B (fig. b)
    ──────────────────
    N1' = N1 (stesso appoggio)
    N2B = (a-av, -c)  testa di C'1 verticale
    N3B = (a-av + z2/tan α, -(H-c))  ancoraggio ferri piegati (nodo fondo)
    N4B = (a-av + 2z2/tan α, -c)     nodo testa C'3
    Aste: C'1 (N1→N2B, verticale), T'1 (N2B→N3B, inclinato basso-destra),
          C'2 (N2B→N4B, orizz. compresso in alto),
          C'3 (N3B→N4B, inclinato alto-destra),
          T'2 (N3B→destra, orizz. teso in basso)
    """
    a_g  = geo['a'];  H_g = geo['H'];  h1_g = geo['h1']
    av_g = geo['av']; c_g = 0.05
    Ved_g = carichi['Ved']

    N1  = ris['N1'];  N3  = ris['N3'];  N4  = ris['N4'];  N2A = ris['N2A']
    N2B = ris['N2B']; N3B = ris['N3B']; N4B = ris['N4B']

    L_tot = max(a_g + H_g + 0.5, N4B[0] + 0.4)

    fig = go.Figure()

    # ── Sagoma calcestruzzo ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[0,    L_tot, L_tot,   a_g,   a_g,   0,    0],
        y=[0,    0,    -H_g,   -H_g, -h1_g, -h1_g,  0],
        fill="toself", fillcolor="rgba(180,180,180,0.20)",
        line=dict(color="gray", width=2),
        hoverinfo="skip", showlegend=False
    ))
    # linea verticale tratteggiata allo step (x=a)
    fig.add_trace(go.Scatter(
        x=[a_g, a_g], y=[-h1_g, -H_g],
        mode="lines", line=dict(color="gray", width=1, dash="dot"),
        hoverinfo="skip", showlegend=False
    ))

    # ── Freccia reazione V_Ed ────────────────────────────────────────────────
    fig.add_annotation(
        x=N1[0], y=N1[1],
        ax=N1[0], ay=N1[1] - 0.32,
        xref='x', yref='y', axref='x', ayref='y',
        text=f"<b>V_Ed={Ved_g:.0f}kN</b>",
        showarrow=True, arrowhead=3, arrowsize=1.4,
        arrowwidth=3, arrowcolor="green", font=dict(color="green", size=11)
    )
    if carichi['Hed'] > 0:
        fig.add_annotation(
            x=N1[0], y=N1[1],
            ax=N1[0] - 0.25, ay=N1[1],
            xref='x', yref='y', axref='x', ayref='y',
            text=f"H_Ed={carichi['Hed']:.0f}",
            showarrow=True, arrowhead=3, arrowsize=1.2,
            arrowwidth=2, arrowcolor="green", font=dict(color="green", size=9)
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # MODELLO A  (fig. 10.72a)
    #
    # Triangolo C1 – T1 – C2  (θ1 = θ2)
    # ─────────────────────────────────────────────────────────────────────────
    #  N_junc_top = apice del triangolo (cima di T2a, fine di C1 e C2)
    #  N1         = nodo appoggio (base sinistra del triangolo)
    #  N2         = nodo base destra del triangolo (fine di T1, piede di C2)
    #
    #  Poiché θ2 = θ1 = arctan(z1/av), il lato orizzontale di C2 = av
    #  → N2.x = N_junc_top.x + av
    # ═══════════════════════════════════════════════════════════════════════════
    x_junc      = a_g + 0.15                        # x nodo di giunzione (trave)
    N_junc_top  = (x_junc,          -c_g)           # apice: testa C1, C2 e T2a
    N_junc_bot  = (x_junc,          -(H_g - c_g))   # piede T2a, inizio T3
    N2_A_tri    = (x_junc + av_g,   N1[1])           # nodo base destra (fine T1, piede C2)

    if k_A > 0:
        # C1 – puntone da N1 (appoggio) → N_junc_top (apice)
        fig.add_trace(go.Scatter(
            x=[N1[0], N_junc_top[0]], y=[N1[1], N_junc_top[1]],
            mode="lines",
            line=dict(color="royalblue", width=4, dash="dash"),
            name=f"C1 = {ris['C1_A']:.1f} kN  (θ1={ris['theta1_deg']:.1f}°)"
        ))

        # C2 – puntone simmetrico: da N2_A_tri → N_junc_top  (θ2 = θ1)
        fig.add_trace(go.Scatter(
            x=[N2_A_tri[0], N_junc_top[0]], y=[N2_A_tri[1], N_junc_top[1]],
            mode="lines",
            line=dict(color="royalblue", width=4, dash="dash"),
            name=f"C2 (θ2=θ1={ris['theta1_deg']:.1f}°)"
        ))

        # T2a – staffe verticali DOPO la risega: da N_junc_top a N_junc_bot
        # (stessa x di C1 e T3 → i tre elementi si toccano ai nodi)
        fig.add_trace(go.Scatter(
            x=[N_junc_top[0], N_junc_bot[0]],
            y=[N_junc_top[1], N_junc_bot[1]],
            mode="lines+markers",
            line=dict(color="darkred", width=3),
            marker=dict(size=8, color="darkred", symbol="circle"),
            name=f"T2a staffe = {ris['T2a']:.1f} kN"
        ))

        # C3 – puntone da fine T1 (N2_A_tri) → inizio T3 (N_junc_bot)
        fig.add_trace(go.Scatter(
            x=[N2_A_tri[0], N_junc_bot[0]], y=[N2_A_tri[1], N_junc_bot[1]],
            mode="lines",
            line=dict(color="royalblue", width=4, dash="dash"),
            name=f"C3"
        ))

        # C4 – puntone 45° nel corpo trave: N_junc_bot → N2A
        fig.add_trace(go.Scatter(
            x=[N_junc_bot[0], N2A[0]], y=[N_junc_bot[1], N2A[1]],
            mode="lines",
            line=dict(color="royalblue", width=3, dash="dash"),
            name=f"C4 = {ris['C4_A']:.1f} kN  (45°)"
        ))

        # Corrente compresso superiore C5: da N_junc_top verso destra
        fig.add_trace(go.Scatter(
            x=[N_junc_top[0], N2A[0]], y=[N_junc_top[1], N_junc_top[1]],
            mode="lines",
            line=dict(color="royalblue", width=2, dash="dot"),
            name="C5 corrente compresso (Mod. A)"
        ))

    # ═══════════════════════════════════════════════════════════════════════════
    # MODELLO B  (fig. 10.72b)
    # ═══════════════════════════════════════════════════════════════════════════
    if k_A < 1.0:
        # C'1 – puntone verticale  N1 → N2B  (direttamente sopra l'appoggio)
        fig.add_trace(go.Scatter(
            x=[N1[0], N2B[0]], y=[N1[1], N2B[1]],
            mode="lines",
            line=dict(color="darkorange", width=3, dash="dash"),
            name=f"C'1 = {ris['Rb']:.1f} kN  (vert.)"
        ))

        # T'1 – ferri piegati  N2B → N3B  (inclinato BASSO-DESTRA a α)
        fig.add_trace(go.Scatter(
            x=[N2B[0], N3B[0]], y=[N2B[1], N3B[1]],
            mode="lines",
            line=dict(color="darkorange", width=4),
            name=f"T'1 ferri piegati = {ris['T1_prime']:.1f} kN  (α={ris['alpha_deg']:.0f}°)"
        ))

        # C'2 – corrente compresso orizzontale superiore  N2B → N4B
        fig.add_trace(go.Scatter(
            x=[N2B[0], N4B[0]], y=[N2B[1], N4B[1]],
            mode="lines",
            line=dict(color="darkorange", width=2, dash="dot"),
            name="C'2 corrente compresso (Mod. B)"
        ))

        # C'3 – puntone diagonale  N3B → N4B  (inclinato ALTO-DESTRA a α)
        fig.add_trace(go.Scatter(
            x=[N3B[0], N4B[0]], y=[N3B[1], N4B[1]],
            mode="lines",
            line=dict(color="darkorange", width=3, dash="dash"),
            name=f"C'3 = {ris['T1_prime']:.1f} kN  (45°)"
        ))

        # T'2 – tirante inferiore trave  N3B → destra
        fig.add_trace(go.Scatter(
            x=[N3B[0], L_tot - 0.05], y=[N3B[1], N3B[1]],
            mode="lines+markers",
            line=dict(color="firebrick", width=4),
            marker=dict(size=8, color="firebrick", symbol="circle"),
            name=f"T'2 trave (Mod. B) = {ris['T2_B']:.1f} kN"
        ))

        # marker nodo N3B
        fig.add_trace(go.Scatter(
            x=[N3B[0]], y=[N3B[1]],
            mode="markers",
            marker=dict(size=11, color="darkorange", symbol="diamond"),
            showlegend=False, hoverinfo="skip"
        ))

    # ── T1 – tirante orizzontale: solo nel Modello A (nel B non esiste)
    if k_A > 0:
        fig.add_trace(go.Scatter(
            x=[N1[0], N2_A_tri[0]], y=[N1[1], N1[1]],
            mode="lines+markers",
            line=dict(color="crimson", width=5),
            marker=dict(size=8, color="crimson", symbol="circle"),
            name=f"T1 = {ris['T1']:.1f} kN"
        ))

    # ── T3 – tirante inferiore trave (Mod. A) ────────────────────────────────
    if k_A > 0:
        # T3 parte dal piede di T2a (N_junc_bot), non dal bordo della risega
        fig.add_trace(go.Scatter(
            x=[N_junc_bot[0], L_tot - 0.05], y=[N_junc_bot[1], N_junc_bot[1]],
            mode="lines+markers",
            line=dict(color="firebrick", width=4),
            marker=dict(size=8, color="firebrick", symbol="circle"),
            name=f"T3 trave (Mod. A) = {ris['T3']:.1f} kN"
        ))

    # ── Punto appoggio ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[N1[0]], y=[N1[1]],
        mode="markers",
        marker=dict(size=13, color="green", symbol="circle-open",
                    line=dict(width=2.5, color="green")),
        name="Nodo appoggio (N1)"
    ))

    # ── Etichette geometriche ────────────────────────────────────────────────
    fig.add_annotation(x=a_g/2, y=-h1_g - 0.12,
                       text=f"a={a_g:.2f}m", showarrow=False,
                       font=dict(size=9, color="gray"))
    fig.add_annotation(x=a_g - av_g/2, y=-h1_g - 0.06,
                       text=f"av={av_g:.2f}m", showarrow=False,
                       font=dict(size=9, color="gray"))
    fig.add_annotation(x=-0.09, y=-h1_g/2,
                       text=f"h1={h1_g:.2f}", showarrow=False,
                       font=dict(size=9, color="gray"))
    fig.add_annotation(x=L_tot + 0.06, y=-H_g/2,
                       text=f"H={H_g:.2f}", showarrow=False,
                       font=dict(size=9, color="gray"))

    fig.update_layout(
        xaxis=dict(title="x [m]", zeroline=False),
        yaxis=dict(title="y [m]", zeroline=False,
                   scaleanchor="x", scaleratio=1,
                   range=[-H_g - 0.55, 0.40]),
        height=540,
        margin=dict(l=30, r=30, t=40, b=20),
        legend=dict(x=0.01, y=0.01,
                    bgcolor="rgba(255,255,255,0.88)",
                    bordercolor="lightgray", borderwidth=1,
                    font=dict(size=11))
    )
    return fig


# ─── TABELLA VERIFICHE ────────────────────────────────────────────────────────
def mostra_verifiche(ris: dict, titolo: str = ""):
    if titolo:
        st.markdown(f"### {titolo}")
    if ris["esito"]:
        st.success("Tutte le verifiche soddisfatte (tiranti + puntoni).")
    else:
        st.error("Una o più verifiche NON soddisfatte (tiranti e/o puntoni).")

    k_A = ris["k_A"]

    # ── Puntoni (verifica σ_c ≤ ν'·fcd) ──────────────────────────────────────
    st.markdown(
        f"**Puntoni (C)** — verifica calcestruzzo: "
        f"ν' = {ris['nu_p']:.3f}, "
        f"f_cd,eff = ν'·f_ck/γ_c = **{ris['fcd_eff']:.2f} MPa**"
    )
    rows_C = []
    if k_A > 0:
        rows_C += [
            ("C1 – puntone dente (incl.)",       ris["C1_A"], ris["ur_C1_A"],
             f"l_s={ris['ls_C1']*100:.1f} cm"),
            ("C2 – puntone simm. (θ2=θ1)",       ris["C2_A"], ris["ur_C2_A"],
             f"l_s={ris['ls_C1']*100:.1f} cm"),
            ("C3 – collegamento T1→T3",           ris["C3_A"], ris["ur_C3_A"],
             f"l_s={ris['l_b']*100:.1f} cm"),
            ("C4 – puntone trave 45°",             ris["C4_A"], ris["ur_C4_A"],
             f"l_s={ris['l_b']*100:.1f} cm"),
        ]
    if k_A < 1.0:
        rows_C += [
            ("C'1 – puntone verticale",            ris["Rb"],   ris["ur_C1_B"],
             f"l_s={ris['l_b']*100:.1f} cm"),
            ("C'2 – compressione orizz. sup.",     ris["C2_B"], ris["ur_C2_B"],
             "l_s=2c"),
            ("C'3 – puntone diagonale (=T'1)",     ris["C3_B"], ris["ur_C3_B"],
             f"l_s={ris['ls_C3B']*100:.1f} cm"),
        ]
    df_C = pd.DataFrame({
        "Puntone":    [r[0] for r in rows_C],
        "Forza [kN]": [r[1] for r in rows_C],
        "UR [%]":     [f"{r[2]*100:.1f}" for r in rows_C],
        "Esito":      ["OK" if r[2] <= 1.0 else "NO" for r in rows_C],
        "Nodo":       [r[3] for r in rows_C],
    })
    st.dataframe(df_C, hide_index=True, use_container_width=True)

    # ── Tiranti (forze di trazione, con verifica UR) ───────────────────────────
    st.markdown("**Tiranti (T)**")
    rows_T = []
    if k_A > 0:
        rows_T += [
            ("T1 – dente (orizz.)",      ris["T1"],      ris["ur_T1"]),
            ("T2a – staffe (vert.)",     ris["T2a"],     ris["ur_T2a"]),
        ]
    if k_A < 1.0:
        rows_T.append(
            ("T'1 – ferri piegati",      ris["T1_prime"], ris["ur_T1_prime"]),
        )
    rows_T.append(
        ("T3 – trave princ.",            ris["T3"],      ris["ur_T3"]),
    )
    df_v = pd.DataFrame({
        "Tirante":    [r[0] for r in rows_T],
        "Forza [kN]": [r[1] for r in rows_T],
        "UR [%]":     [f"{r[2]*100:.1f}" for r in rows_T],
        "Esito":      ["OK" if r[2] <= 1.0 else "NO" for r in rows_T],
    })
    st.dataframe(df_v, hide_index=True, use_container_width=True)


# ─── TABS ─────────────────────────────────────────────────────────────────────
st.divider()
tab1, tab2, tab3 = st.tabs([
    "Modello A – Solo Staffe",
    "Modello B – Solo Ferri Piegati",
    "Modello Combinato A+B"
])

# ── TAB 1 : Modello A ─────────────────────────────────────────────────────────
with tab1:
    ris_A = analizza_modello_combinato(
        geo, carichi, mat, A_T1, A_T2a, A_T1_prime, alpha, A_T3, k_A=1.0)
    st.markdown(
        f"**Puntone C1** a θ₁ = **{ris_A['theta1_deg']:.1f}°**  |  "
        f"C1 = **{ris_A['C1_A']:.1f} kN**  |  "
        f"Puntone trave C4 = **{ris_A['C4_A']:.1f} kN** (45°)"
    )
    col_g, col_v = st.columns([3, 2])
    with col_g:
        st.plotly_chart(disegna_traliccio(ris_A, geo, carichi, 1.0),
                        use_container_width=True)
    with col_v:
        mostra_verifiche(ris_A, "Verifiche Modello A")
        with st.expander("Equazioni Modello A (Prospetto 10.15)", expanded=False):
            st.markdown("**Angolo e puntoni**")
            st.latex(
                rf"\theta_1 = \arctan\!\left(\frac{{z_1}}{{a_v}}\right)"
                rf"= \arctan\!\left(\frac{{{ris_A['z1']:.3f}}}{{{av:.3f}}}\right)"
                rf"= {ris_A['theta1_deg']:.1f}^\circ")
            st.latex(
                rf"C_1 = \frac{{R_a}}{{\sin\theta_1}}"
                rf"= {ris_A['C1_A']:.1f}\ \mathrm{{kN}}")
            st.latex(
                rf"C_2 = C_1 = {ris_A['C2_A']:.1f}\ \mathrm{{kN}}"
                rf"\quad (\theta_2=\theta_1)")
            st.latex(
                rf"L_{{C3}} = \sqrt{{a_v^2+(H-h_1)^2}}"
                rf"= {np.sqrt(av**2+(H-h1)**2):.3f}\ \mathrm{{m}}")
            st.latex(
                rf"C_3 = R_a \cdot \frac{{L_{{C3}}}}{{H-h_1}}"
                rf"= {ris_A['C3_A']:.1f}\ \mathrm{{kN}}")
            st.latex(
                rf"C_4 = R_a\sqrt{{2}} = {ris_A['C4_A']:.1f}\ \mathrm{{kN}}"
                rf"\quad (45°)")
            st.markdown("**Tiranti**")
            st.latex(
                rf"T_1 = \frac{{R_a \cdot a_v}}{{z_1}} + H_{{Ed}}"
                rf"= {ris_A['T1_A']:.1f} + {Hed:.1f} = {ris_A['T1']:.1f}\ \mathrm{{kN}}")
            st.latex(rf"T_{{2a}} = R_a = {ris_A['T2a']:.1f}\ \mathrm{{kN}}")
            st.latex(
                rf"T_3 = R_a\!\left(1+\frac{{a_v}}{{2z_1}}\right) + H_{{Ed}}"
                rf"= {ris_A['T3']:.1f}\ \mathrm{{kN}}")
            st.markdown("**Verifica puntoni** — σ_c = C/(b·l_s) ≤ ν'·f_cd")
            st.latex(
                rf"\nu' = 0.6\!\left(1-\frac{{f_{{ck}}}}{{250}}\right)"
                rf"= {ris_A['nu_p']:.3f}")
            st.latex(
                rf"f_{{cd,eff}} = \nu'\!\cdot\!\frac{{f_{{ck}}}}{{\gamma_c}}"
                rf"= {ris_A['fcd_eff']:.2f}\ \mathrm{{MPa}}")
            st.latex(
                rf"l_{{s,C1}} = l_b\sin\theta_1 + 2c\cos\theta_1"
                rf"= {ris_A['ls_C1']*100:.1f}\ \mathrm{{cm}}")

# ── TAB 2 : Modello B ─────────────────────────────────────────────────────────
with tab2:
    ris_B = analizza_modello_combinato(
        geo, carichi, mat, A_T1, A_T2a, A_T1_prime, alpha, A_T3, k_A=0.0)
    st.markdown(
        f"**C'1** verticale = **{ris_B['Rb']:.1f} kN**  |  "
        f"**T'1** ferri piegati = **{ris_B['T1_prime']:.1f} kN** (α={alpha:.0f}°)  |  "
        f"**T'2** trave = **{ris_B['T2_B']:.1f} kN**"
    )
    st.info(
        "Angotti: il solo Modello B lascia il bordo inferiore del dente privo di armatura. "
        "Raccomandato: almeno 50% del taglio al Modello A."
    )
    col_g, col_v = st.columns([3, 2])
    with col_g:
        st.plotly_chart(disegna_traliccio(ris_B, geo, carichi, 0.0),
                        use_container_width=True)
    with col_v:
        mostra_verifiche(ris_B, "Verifiche Modello B")
        with st.expander("Equazioni Modello B (Prospetto 10.16)", expanded=False):
            st.markdown("**Puntoni**")
            st.latex(rf"C'_1 = R_b = {ris_B['Rb']:.1f}\ \mathrm{{kN}}")
            st.latex(
                rf"C'_2 = \frac{{R_b}}{{\tan\alpha}}"
                rf"= {ris_B['C2_B']:.1f}\ \mathrm{{kN}}")
            st.latex(
                rf"C'_3 = \frac{{R_b}}{{\sin\alpha}}"
                rf"= {ris_B['C3_B']:.1f}\ \mathrm{{kN}}")
            st.markdown("**Tiranti**")
            st.latex(
                rf"T'_1 = \frac{{R_b}}{{\sin\alpha}}"
                rf"= \frac{{{ris_B['Rb']:.1f}}}{{\sin {alpha:.0f}^\circ}}"
                rf"= {ris_B['T1_prime']:.1f}\ \mathrm{{kN}}")
            st.latex(
                rf"T'_2 = \frac{{2\,R_b}}{{\tan\alpha}}"
                rf"= {ris_B['T2_B']:.1f}\ \mathrm{{kN}}")
            st.caption(
                "C'1 è verticale → nessuna componente orizzontale all'appoggio "
                "(T1 dente = solo H_Ed)."
            )
            st.markdown("**Verifica puntoni** — σ_c = C/(b·l_s) ≤ ν'·f_cd")
            st.latex(
                rf"\nu' = 0.6\!\left(1-\frac{{f_{{ck}}}}{{250}}\right)"
                rf"= {ris_B['nu_p']:.3f}")
            st.latex(
                rf"f_{{cd,eff}} = \nu'\!\cdot\!\frac{{f_{{ck}}}}{{\gamma_c}}"
                rf"= {ris_B['fcd_eff']:.2f}\ \mathrm{{MPa}}")
            st.latex(
                rf"l_{{s,C'1}} = l_b = {ris_B['l_b']*100:.1f}\ \mathrm{{cm}}"
                rf"\quad\quad"
                rf"l_{{s,C'3}} = l_b\sin\alpha+2c\cos\alpha"
                rf"= {ris_B['ls_C3B']*100:.1f}\ \mathrm{{cm}}")

# ── TAB 3 : Modello Combinato ─────────────────────────────────────────────────
with tab3:
    st.markdown(
        "Assegna la percentuale di V_Ed al **Modello A** (staffe). "
        "Il rimanente va al **Modello B** (ferri piegati).")
    k_perc = st.slider("% Taglio → Modello A  (min. 50% consigliato)",
                       0, 100, 50, 5)
    k_A_comb = k_perc / 100.0
    ris_C = analizza_modello_combinato(
        geo, carichi, mat, A_T1, A_T2a, A_T1_prime, alpha, A_T3, k_A_comb)

    col_g, col_v = st.columns([3, 2])
    with col_g:
        fig_C = disegna_traliccio(ris_C, geo, carichi, k_A_comb)
        st.plotly_chart(fig_C, use_container_width=True)
    with col_v:
        mostra_verifiche(ris_C, f"Verifiche ({k_perc}% A + {100-k_perc}% B)")
        st.markdown("#### Contributi per modello")
        st.markdown(
            f"| Grandezza | Mod. A | Mod. B | **Totale** |\n"
            f"|-----------|-------:|-------:|-----------:|\n"
            f"| Quota [kN] | {ris_C['Ra']:.1f} | {ris_C['Rb']:.1f} | {Ved:.1f} |\n"
            f"| **C1 / C'1** | {ris_C['C1_A']:.1f} | {ris_C['Rb']:.1f} | — |\n"
            f"| **C2** | {ris_C['C2_A']:.1f} | {ris_C['C2_B']:.1f} | — |\n"
            f"| **C3** | {ris_C['C3_A']:.1f} | {ris_C['C3_B']:.1f} | — |\n"
            f"| **C4** | {ris_C['C4_A']:.1f} | — | — |\n"
            f"| T1 dente | {ris_C['T1_A']:.1f} | — | **{ris_C['T1']:.1f}** |\n"
            f"| T2a / T'1 | {ris_C['T2a']:.1f} | {ris_C['T1_prime']:.1f} | — |\n"
            f"| T3 trave | {ris_C['T2_A']:.1f} | {ris_C['T2_B']:.1f} | **{ris_C['T3']:.1f}** |\n"
        )
        st.latex(
            rf"T_1 = \frac{{R_a \cdot a_v}}{{z_1}} + H_{{Ed}}"
            rf"= {ris_C['T1_A']:.1f} + {Hed:.1f} = {ris_C['T1']:.1f}\ \mathrm{{kN}}")
        st.latex(
            rf"T_3 = R_a\!\left(1+\frac{{a_v}}{{2z_1}}\right)"
            rf"+ \frac{{2R_b}}{{\tan\alpha}} + H_{{Ed}}"
            rf"= {ris_C['T3']:.1f}\ \mathrm{{kN}}")

# ─── PDF ──────────────────────────────────────────────────────────────────────
st.divider()
if st.button("Genera PDF (Modello Combinato)"):
    try:
        _ris_pdf = ris_C
    except NameError:
        _ris_pdf = analizza_modello_combinato(
            geo, carichi, mat, A_T1, A_T2a, A_T1_prime, alpha, A_T3, 0.5)
    img_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            fig_C.write_image(tmp.name, engine="kaleido")
            img_path = tmp.name
    except Exception:
        pass
    pdf_bytes = genera_pdf_report(_ris_pdf, img_path)
    st.download_button("Scarica PDF", data=pdf_bytes,
                       file_name="sella_gerber.pdf",
                       mime="application/pdf", type="primary")
    if img_path and os.path.exists(img_path):
        os.remove(img_path)
