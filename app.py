"""Algorithmische Kollusion: lernen zwei Q-Agenten sich zu einigen? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Neuntes Stück der Wiederholte-Spiele-Linie: Zwei Preis-Algorithmen lernen allein aus Erfahrung, ohne jede Absprache. Landen sie bei überhöhten Preisen, und hält eine
gelernte Vergeltung sie dort?

Lauffähig mit: streamlit run app.py
"""

import time

import streamlit as st

import akl_constants as C
import akl_evaluation as E
from akl_evaluation import Settings, fmt, fmt_int
from akl_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from akl_visualization import build_deviation_chart, build_deviation_gains, build_learning_curve, build_spread

st.set_page_config(page_title="Algorithmische Kollusion: lernen zwei Q-Agenten sich zu einigen? – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return E.analyse(settings)


st.title("🤖 Algorithmische Kollusion: lernen zwei Q-Agenten sich zu einigen?")
st.markdown(
    """
Bisher saßen Spieler mit festen Strategien am Tisch. Hier **lernen** sie: Zwei Preis-Algorithmen setzen jede Woche den Preis für eine Frachtrelation, sehen nur ihren
eigenen Gewinn und probieren aus. Sie sprechen sich nie ab, sie tauschen keine Nachricht. Trotzdem können sie bei Preisen **oberhalb** des Wettbewerbspreises landen,
und das haben Calvano, Calzolari, Denicolò und Pastorello (2020) in Simulationen gezeigt. Diese Demo baut dasselbe Modell nach und misst, was davon trägt.

Es ist ein **Simulationsbefund unter bestimmten Annahmen**, keine Aussage über reale Märkte und keine rechtliche Bewertung.

Weiter unten: **❓ Kernfrage**, vier Experimente (**🔬**, aus einer vorgerechneten Tabelle über je 20 Zufalls-Seeds), **🚧 Wo die Annahmen enden** und der Expander
**📐 Mathematische Formulierung**. Vorgänger in der Linie sind [green-porter-demo](https://sebastianhanisch-green-porter-demo.streamlit.app/) (Preiskriege im
Gleichgewicht) und [folk-theorem-demo](https://sebastianhanisch-folk-theorem-demo.streamlit.app/) (was Wiederholung tragen kann); das Lernverfahren steht in
[q-learning-demo](https://sebastianhanisch-q-learning-demo.streamlit.app/).
"""
)

with st.expander("So funktioniert der Markt", expanded=True):
    st.markdown(
        """
1. **Zwei Frachtanbieter** bieten dieselbe Relation an, dazu gibt es ein Außenangebot (nicht buchen). Die Nachfrage fällt nach einer **Logit-Regel**: Wer billiger ist,
   bekommt mehr Aufträge, aber nie alle.
2. **Jede Periode** setzt jeder einen Preis aus einem Raster von 15 Preisen. Die Grenzkosten sind für beide 1,00.
3. **Zwei Referenzpreise:** der **Nash-Preis** (beide konkurrieren, jeder macht das Beste aus dem Preis des anderen) und der **Monopolpreis** (ein gemeinsamer Preis,
   der die Summe beider Gewinne maximiert). Das Raster reicht etwas über beide hinaus.
4. **Kollusionsindex Δ:** 0 heißt Nash-Gewinn, 1 heißt gemeinsamer Monopolgewinn. Alles dazwischen ist „teilweise überhöht“.
        """
    )

with st.expander("Die beiden Lernenden: wer macht was?", expanded=True):
    st.markdown(
        "| Baustein | Bedeutung | Regler |\n|---|---|---|\n"
        "| **Q-Lernen** | Jeder Agent führt eine Tabelle Q(Zustand, Preis): den geschätzten Wert, in diesem Zustand diesen Preis zu setzen. Nach jeder Periode wird der Eintrag um α "
        "in Richtung „Gewinn heute + γ · bester Wert morgen“ verschoben. | α, γ |\n"
        "| **Erkundung** | Mit Wahrscheinlichkeit ε probiert ein Agent einen zufälligen Preis, sonst den besten bekannten. ε fällt von 1 mit der Zeit gegen 0 (ε = e^(−β·t)). | β |\n"
        "| **Gedächtnis** | Der Zustand ist der letzte Preis beider (15 × 15 Zustände). Ohne Gedächtnis gibt es nur einen Zustand: Der Agent kann nicht auf den letzten Preis des anderen "
        "reagieren. | Gedächtnis |\n"
        "| **Voraussicht γ** | Wie stark spätere Gewinne zählen. γ = 0: nur die laufende Periode. | γ |"
    )
    st.caption("Beide Agenten lernen unabhängig voneinander und kennen weder die Nachfrage noch die Strategie des anderen. Der Lauf endet, wenn sich die gierige Strategie "
               f"{fmt_int(C.STABLE_PERIODS)} Perioden lang nicht mehr ändert.")

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Lernen**")
    alpha = st.slider("Lernrate α", *bounds("alpha_slider"), step=C.ALPHA_STEP, key="alpha_slider", help="Wie weit ein Q-Wert pro Erfahrung in Richtung der neuen Schätzung wandert.")
    gamma = st.slider("Voraussicht γ", *bounds("gamma_slider"), step=C.GAMMA_STEP, key="gamma_slider", help="Gewicht späterer Gewinne gegenüber dem heutigen.")
    beta = st.slider("Erkundungsabfall β (in 10⁻⁵ je Periode)", *bounds("beta_slider"), step=C.BETA_STEP, key="beta_slider",
                     help="Je größer, desto schneller hört das Ausprobieren auf. Der Standard ist 2·10⁻⁵; Calvano et al. nutzen 0,4·10⁻⁵ (steht in der Tabelle unten).")
    st.markdown("**Gedächtnis**")
    memory = st.selectbox("Zustandsgedächtnis", C.MEMORY_KEYS, key="memory_select", format_func=lambda k: C.MEMORY_LABELS[k],
                          help="Was ein Agent vom letzten Zug sieht.")
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), step=1, key="seed_input", help="Legt fest, welche Zufallsentscheidungen im Lauf fallen.")
    st.button("🎲 Anderen Zufalls-Seed würfeln", on_click=randomize_seed, key="seed_button")

settings = Settings(round(float(alpha), 2), int(beta), round(float(gamma), 2), memory, int(seed))
sync_query_params({"alpha_slider": settings.alpha, "beta_slider": settings.beta, "gamma_slider": settings.gamma, "memory_select": settings.memory, "seed_input": settings.seed})

t0 = time.time()
with st.spinner("Zwei Lernende üben, das dauert einige Sekunden (bis zu 2,5 Mio. Perioden) ..."):
    an = _analysis(settings)
elapsed = time.time() - t0
mk = an.market
r = an.result

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎬 Ein Lernlauf")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Mittlerer Preis im Dauerzustand", fmt(r.avg_price), delta=f"Nash {fmt(mk.nash_price)} · Monopol {fmt(mk.monopoly_price)}", delta_color="off")
m2.metric("Kollusionsindex Δ", fmt(r.delta), delta=E.level_of(r.delta), delta_color="off")
m3.metric("Perioden bis Konvergenz", fmt_int(r.periods), delta="konvergiert" if r.converged else "Obergrenze erreicht", delta_color="off")
m4.metric("Länge des Preiszyklus", f"{len(r.cycle_states)}", delta="1 = fester Preis", delta_color="off")
st.plotly_chart(build_learning_curve(r.trace, mk.nash_price, mk.monopoly_price), width="stretch", key="learning_chart")
st.caption("Zu Beginn probieren beide fast nur zufällig aus (ε nahe 1), der Durchschnitt liegt in der Rastermitte. Mit fallender Erkundung setzen sich die gelernten Preise durch.")
cyc = " → ".join(f"({fmt(a)}; {fmt(b)})" for a, b in r.cycle_prices)
st.caption(f"Dauerzustand (Preis Anbieter 1; Preis Anbieter 2), Rechenzeit {fmt(elapsed, 1)} s: {cyc}")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Lernen zwei Q-Agenten überhöhte Preise, und trägt die Vergeltung?")
level, sentence = E.verdict(an, settings)
{"success": st.success, "warning": st.warning, "info": st.info}[level](sentence)

if an.deviation is not None:
    st.markdown("### Abweichungstest")
    st.caption("Im Dauerzustand wird Anbieter 1 für **eine** Periode auf den Rasterpreis nächst dem Nash-Preis gezwungen, danach spielen beide wieder, wie sie gelernt haben.")
    st.plotly_chart(build_deviation_chart(an.deviation, mk.nash_price, mk.monopoly_price), width="stretch", key="deviation_chart")
    ret = an.deviation.return_period
    st.caption("Der Rivale kehrt " + (f"ab Periode {ret} dauerhaft zu seinem Basispreis zurück." if ret is not None else
                                      f"innerhalb von {C.DEVIATION_HORIZON} Perioden nicht dauerhaft zu seinem Basispreis zurück."))
else:
    st.info("Ohne Gedächtnis gibt es keinen Abweichungstest: Der Rivale sieht den Zustand nicht und kann nicht gezielt bestrafen.")
st.markdown("---")

# --- Experimente (vorgerechnet) ------------------------------------------------------------------------------------------------------------

sweeps = E.load_sweeps()
if sweeps is None:
    st.warning("Die vorgerechnete Tabelle `akl_sweeps.json` fehlt. Erzeugen mit `python akl_make_sweeps.py`.")
else:
    n_seeds = len(sweeps["meta"]["seeds"])

    st.subheader("🔬 Experiment: Zählt die Zukunft? (Voraussicht γ)")
    st.caption(f"Je Punkt {n_seeds} Läufe mit festen Zufalls-Seeds, übrige Regler auf der Voreinstellung. Vorgerechnet, nicht live.")
    series = E.sweep_series("gamma")
    st.plotly_chart(build_spread([{"x": e["x"], "runs": e["config"]["runs"]} for e in series], "Voraussicht γ"), width="stretch", key="gamma_chart")
    lo, hi = series[0], series[-1]
    gn = " und ".join(fmt(g[2]) for g in sorted(mk.grid_nash, key=lambda g: g[2]))
    st.info(f"Bei γ = {fmt(lo['x'])} liegt der mediane Index bei {fmt(lo['summary']['median'])}, bei γ = {fmt(hi['x'])} bei {fmt(hi['summary']['median'])}; "
            f"dazwischen steigt er mit γ. Zum Vergleich: Das Einmalspiel hat auf dem Raster zwei reine Nash-Gleichgewichte mit Δ = {gn}, ein Index in dieser Gegend ist also noch "
            "Wettbewerb.")

    st.subheader("🔬 Experiment: Braucht es Gedächtnis?")
    both = E.find_config(C.DEFAULT_ALPHA, float(C.DEFAULT_BETA), C.DEFAULT_GAMMA, "both")
    none = E.find_config(C.DEFAULT_ALPHA, float(C.DEFAULT_BETA), C.DEFAULT_GAMMA, "none")
    if both and none:
        sb, sn = E.summarize(both), E.summarize(none)
        st.markdown(
            "| Gedächtnis | Median Δ | kleinster / größter Wert | konvergiert | Perioden (Median) |\n|---|---|---|---|---|\n"
            f"| {C.MEMORY_LABELS['both']} | {fmt(sb['median'])} | {fmt(sb['min'])} / {fmt(sb['max'])} | {sb['converged']} von {sb['n']} | {fmt_int(sb['periods_median'])} |\n"
            + f"| {C.MEMORY_LABELS['none']} | {fmt(sn['median'])} | {fmt(sn['min'])} / {fmt(sn['max'])} | {sn['converged']} von {sn['n']} | {fmt_int(sn['periods_median'])} |"
        )
        st.info(f"Ohne Gedächtnis liegt der Index im Median bei {fmt(sn['median'])}, mit Gedächtnis bei {fmt(sb['median'])}. "
                "Überhöhte Preise brauchen also keine gelernte Vergeltung: Auch Agenten, die den letzten Preis des anderen gar nicht sehen, landen oberhalb des Nash-Preises.")

    st.subheader("🔬 Experiment: Wie wichtig sind Lernrate und Erkundung?")
    col_a, col_b = st.columns(2)
    with col_a:
        sa = E.sweep_series("alpha")
        st.plotly_chart(build_spread([{"x": e["x"], "runs": e["config"]["runs"]} for e in sa], "Lernrate α", title="Δ über die Lernrate α"), width="stretch", key="alpha_chart")
    with col_b:
        sbeta = E.sweep_series("beta")
        st.plotly_chart(build_spread([{"x": e["x"], "runs": e["config"]["runs"]} for e in sbeta], "Erkundungsabfall β (10⁻⁵)", title="Δ über den Erkundungsabfall β", log_x=True),
                        width="stretch", key="beta_chart")
    am = [e["summary"]["median"] for e in sa]
    st.info(f"Lernrate: Der mediane Index liegt zwischen {fmt(min(am))} und {fmt(max(am))}, ohne einfachen Trend. Erkundung: bei β = {fmt(sbeta[0]['x'], 1)}·10⁻⁵ "
            f"(Calvano et al.) {fmt(sbeta[0]['summary']['median'])} mit im Median {fmt_int(sbeta[0]['summary']['periods_median'])} Perioden, bei β = {fmt(sbeta[-1]['x'], 0)}·10⁻⁵ "
            f"{fmt(sbeta[-1]['summary']['median'])} mit {fmt_int(sbeta[-1]['summary']['periods_median'])} Perioden. Die langsamste Erkundung landet im Median höher als die schnellste, braucht aber ein Vielfaches der Perioden.")

    st.subheader("🔬 Experiment: Hält die Vergeltung?")
    if both:
        sd = E.summarize(both)
        st.plotly_chart(build_deviation_gains(both["runs"]), width="stretch", key="devgain_chart")
        st.info(f"In der Voreinstellung lohnt sich die einmalige Abweichung auf den Nash-Preis in {sd['dev_n'] - sd['dev_unprofitable']} von {sd['dev_n']} Läufen (diskontierter Gewinn "
                f"über {C.DEVIATION_HORIZON} Perioden über null); in {sd['dev_unprofitable']} von {sd['dev_n']} trägt die gelernte Strafe. Der Rivale senkt seinen Preis in "
                f"{sd['dev_rival_cuts']} von {sd['dev_n']} Läufen.")
st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Logit-Nachfrage, zwei gleiche Anbieter, Preisraster mit 15 Stufen** | Mehr Anbieter, eine andere Nachfrage oder ein feineres Raster können die Höhe der Preise ändern; hier nicht gerechnet. | – |
| **Simulation statt Felddaten** | Der Befund gilt für diese Lernregel und diese Parameter; er sagt nichts darüber, ob reale Preis-Software so lernt oder ob reale Märkte so aussehen. | – |
| **Tabellarisches Q-Lernen mit festen Parametern** | Andere Lernverfahren (neuronale Netze, Bandit-Verfahren) oder Parameter geben andere Ergebnisse; der Befund hängt sichtbar an α, β, γ. | `q-learning-demo`, `bandit-demo` |
| **Beide lernen mit demselben Verfahren und synchron** | Ein Anbieter mit Wissen über das Modell oder ein Mensch im Markt verändert die Dynamik. | – |
| **Der Abweichungstest prüft eine einzelne Senkung** | Andere Abweichungen und längere Horizonte können anders ausgehen; gelernte Strategien sind nur entlang des Dauerzustands geprüft. | `folk-theorem-demo` |
| **Kein Rechtsurteil** | „Kollusion“ ist hier ein Modellbegriff für überhöhte Gleichgewichtspreise ohne Absprache, keine Aussage zum Kartellrecht. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Nachfrage.** Anbieter $i$ mit Preis $p_i$ hat den Anteil $q_i = \dfrac{e^{(a - p_i)/\mu}}{\sum_j e^{(a - p_j)/\mu} + e^{a_0/\mu}}$ und den Gewinn
$\pi_i = (p_i - c)\,q_i$ mit $a = 2$, $a_0 = 0$, $\mu = 0{,}25$, $c = 1$.

**Referenzpreise.** Nash: $p_N = c + \mu / (1 - q(p_N, p_N))$ (symmetrische Bedingung erster Ordnung, hier numerisch gelöst). Monopol: $p_M = \arg\max_p\,[\pi_1(p,p)+\pi_2(p,p)]$.
Raster: 15 gleich verteilte Preise zwischen $p_N - 0{,}1\,(p_M - p_N)$ und $p_M + 0{,}1\,(p_M - p_N)$.

**Q-Lernen.** Zustand $s$ = beide Preise der Vorperiode. Update:
$Q(s,a) \leftarrow (1-\alpha)\,Q(s,a) + \alpha\,[\,\pi + \gamma \max_{a'} Q(s',a')\,]$, Verhalten $\varepsilon$-gierig mit $\varepsilon_t = e^{-\beta t}$.
Anfangswert $Q_0(a) = \sum_b \pi(a,b) / ((1-\gamma)\,m)$.

**Kollusionsindex.** $\Delta = (\bar\pi - \pi_N)/(\pi_M - \pi_N)$ mit dem mittleren Gewinn $\bar\pi$ im Dauerzustand.

**Abweichungstest.** Ab einem Zustand im Dauerzyklus wird Anbieter 1 in Periode 0 auf den Rasterpreis nächst $p_N$ gezwungen, danach folgen beide ihrer gierigen
Strategie. Bewertet wird $\sum_{t<25}\gamma^t \pi_1^{\mathrm{Abw.}}(t) - \sum_{t<25}\gamma^t \pi_1^{\mathrm{Basis}}(t)$.

**Literatur.** Calvano, Calzolari, Denicolò und Pastorello (2020), *Artificial Intelligence, Algorithmic Pricing, and Collusion*, American Economic Review 110, 3267–3297
(das nachgebaute Modell); dieselben (2021), *Algorithmic collusion with imperfect monitoring*, International Journal of Industrial Organization; Waltman und Kaymak (2008),
*Q-learning agents in a Cournot oligopoly model*, Journal of Economic Dynamics and Control 32, 3275–3293 (Q-Lernen in einem Cournot-Modell, Gedächtnis und Erkundung als Einflussgrößen); Watkins und Dayan
(1992), *Q-learning*, Machine Learning 8, 279–292. Nur Zusammenfassungen gelesen. Die Zahlen hier sind gerechnet, nicht aus den Aufsätzen übernommen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
