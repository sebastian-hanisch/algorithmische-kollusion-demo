# Algorithmische Kollusion: lernen zwei Q-Agenten sich zu einigen? – Streamlit-Demo

<!-- Nach dem Deploy direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-algorithmische-kollusion-demo.streamlit.app/)** -->

Neuntes und letztes Kernstück der **Wiederholte-Spiele-Linie** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“.
Hier fließen die Stücke zusammen: Was Wiederholung tragen kann ([folk-theorem-demo](https://sebastianhanisch-folk-theorem-demo.streamlit.app/)), Strafen bei Beobachtung
([green-porter-demo](https://sebastianhanisch-green-porter-demo.streamlit.app/)) und das Lernen aus Erfahrung
([q-learning-demo](https://sebastianhanisch-q-learning-demo.streamlit.app/)). Zwei **Preis-Algorithmen** lernen ohne jede Absprache, und die Demo misst, wo sie landen.

Vehikel: zwei Frachtanbieter setzen jede Woche den Preis für dieselbe Relation (Logit-Nachfrage mit Außenangebot). Das Modell ist das von Calvano, Calzolari, Denicolò und
Pastorello (2020), mit eigener, nachgebauter Implementierung.

**Das ist ein Simulationsbefund unter bestimmten Annahmen.** Er sagt nichts über reale Märkte oder reale Preis-Software, und „Kollusion“ ist hier ein Modellbegriff für
überhöhte Preise ohne Absprache, keine rechtliche Bewertung.

## Warum dieses Problem

Preis-Software setzt in vielen Märkten Preise automatisch. Wenn zwei solche Programme aus Erfahrung lernen, ohne voneinander zu wissen, können sie bei Preisen landen, die über
dem Wettbewerbspreis liegen. Interessant ist, **was** diese Preise hält: eine gelernte Vergeltung wie in den Stücken davor, oder etwas Einfacheres.

## Modell

Logit-Nachfrage (Qualität 2, Außenangebot 0, Preissensitivität 0,25, Grenzkosten 1), 15 Preise auf einem Raster, das etwas über Nash- und Monopolpreis hinausreicht.
Zwei unabhängige tabellarische Q-Lernende, Zustand = die beiden letzten Preise, Erkundung ε = e^(−β·t). Kennzahl: **Kollusionsindex** Δ = (π̄ − π_N) / (π_M − π_N).

## Methodik

- **Markt:** `shares`, `profits`, `nash_price`, `monopoly_price`, `price_grid` (`akl_algorithm.py`).
- **Lernen:** `learn` (zwei Q-Lernende, Konvergenz, wenn die gierige Strategie 100.000 Perioden unverändert bleibt), `td_update`, `epsilon`, `initial_q`.
- **Dauerzustand:** `cycle_from` bestimmt den Zyklus der gierigen Strategien, daraus Durchschnittspreis, -gewinn und Δ.
- **Abweichungstest:** `deviation_test` zwingt Anbieter 1 einmal auf den Rasterpreis nächst dem Nash-Preis und bewertet 25 Perioden diskontiert.
- **Vorgerechnete Tabelle:** `akl_make_sweeps.py` rechnet je Konfiguration 20 feste Zufalls-Seeds (13 Konfigurationen, 260 Läufe, rund 2 Minuten parallel) nach `akl_sweeps.json`. Die App liest nur.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: 20 feste Zufalls-Seeds je Konfiguration, Voreinstellung α = 0,15, β = 2·10⁻⁵, γ = 0,95, Gedächtnis der letzten beiden Preise.

| Frage | Befund | Test |
|---|---|---|
| Referenzpreise | Nash-Preis 1,473, Monopolpreis 1,925 (stimmt mit der Literatur überein). Das Einmalspiel hat auf dem Raster zwei reine Nash-Gleichgewichte mit Δ = −0,02 und 0,12. | `test_reference_prices_and_grid_nash_indices` |
| Landen sie über dem Wettbewerbspreis? | Ja: In der Voreinstellung liegt der mediane Index bei **0,79** (kleinster 0,61, größter 0,95), alle 20 Läufe konvergieren (im Median 613.679 Perioden), alle 20 liegen über 0,5. | `test_default_configuration` |
| Hält eine gelernte Vergeltung? | Teils: Eine einmalige Abweichung auf den Nash-Preis lohnt sich in **2 von 20** Läufen (diskontiert über 25 Perioden), in 18 trägt die Strafe. Der Rivale senkt seinen Preis in 18 von 20 Läufen; im Median kehrt er nach 5 Perioden zum Basispfad zurück, in 9 Läufen innerhalb von 25 Perioden nicht Periode für Periode (das schließt eine verschobene Phase eines Preiszyklus ein). Größter Verlust der Abweichung 0,408, größter Gewinn 0,023. | `test_deviation_statistics_in_the_default_configuration` |
| Braucht es Gedächtnis? | **Nein.** Ohne Gedächtnis liegt der mediane Index bei **0,95** (0,87 bis 0,99), höher als mit Gedächtnis (0,79); alle 20 Läufe enden bei einem festen Preis. Ein Abweichungstest ist dort nicht definiert, weil der Rivale den Zustand nicht sieht. | `test_without_memory_every_run_ends_at_a_single_price_and_no_deviation_test_exists` |
| Zählt die Zukunft (γ)? | Der mediane Index steigt mit γ: 0,21 (γ = 0), 0,29 (0,25), 0,38 (0,5), 0,60 (0,75), 0,79 (0,95). Bei γ = 0 trägt die Strafe nur in 3 von 20 Läufen. | `test_median_index_rises_with_the_shadow_of_the_future` |
| Erkundung und Lernrate | β = 0,4·10⁻⁵ (Calvano et al.): mediane Δ 0,93 nach im Median 1.804.668 Perioden; β = 10·10⁻⁵: 0,70; β = 20·10⁻⁵: 0,71 nach 149.808 Perioden. Lernrate α von 0,05 bis 0,5: mediane Δ 0,71, 0,78, 0,79, 0,77, 0,72, ohne einfachen Trend. | `test_exploration_decay_and_learning_rate` |

### Befunde gegenüber dem Plan

Der Plan erwartete: zwei Q-Lernende landen zwischen Nash und Monopol, eine erzwungene Preissenkung löst eine Strafphase mit schrittweiser Rückkehr aus. Das bestätigt sich
für die Preishöhe und weitgehend für die Strafe (18 von 20). Zwei Ergänzungen aus der Messung: **Erstens sind die hohen Preise nicht an die Vergeltung gebunden.** Ohne jedes
Gedächtnis landen die Lernenden sogar höher, und ohne Gedächtnis kann es keine gezielte Strafe geben. **Zweitens zählt die Voraussicht:** Bei γ = 0 sinkt der Index auf 0,21.
Auch dieser Wert liegt über dem Nash-Gleichgewicht des Rasters (0,12), das heißt, ein kurzsichtiges Lernen ist auf diesem Raster nicht genau Wettbewerb. Eine Erklärung
dafür wurde nicht weiter untersucht.

## Ehrliche Grenzen

- **Ein Modell, ein Raster, zwei Anbieter:** Mehr Anbieter, andere Nachfrage oder ein feineres Raster sind nicht gerechnet.
- **Simulation, keine Felddaten:** Aussagen gelten für diese Lernregel und diese Parameter.
- **Abweichungstest nur für eine Senkung:** gelernte Strategien sind nur entlang des Dauerzustands geprüft, nicht überall (Q-Werte unbesuchter Zustände bleiben veraltet).
- **Deterministische Zufalls-Seeds, aber nur 20 je Konfiguration:** die Bereiche zeigen Streuung, keine Konfidenzintervalle.
- **Literatur nur über Zusammenfassungen gelesen.** Die Zahlen aus den Aufsätzen wurden nicht übernommen, sondern hier neu gerechnet.

## Verifikation

- **Kern:** je Einheit eine Handrechnung (Logit-Anteile bei gleichen Preisen, Raster, TD-Update, Erkundungsplan, Anfangs-Q, Index, Zyklus, Abweichungstest mit und ohne Strafe).
- **Orakel** (`tests/test_oracle_algorithmische_kollusion.py`): (1) Nash- und Monopolpreis gegen `scipy` (Nullstelle der abgeleiteten Gewinnfunktion, beschränkte
  Minimierung); (2) Gewinntabelle gegen eine direkte numpy-Rechnung; (3) Ein-Agenten-Grenzfall: gegen einen festen Rivalen lernt der Agent die beste Antwort aus
  der Aufzählung; (4) die Lernschleife gegen eine zweite, schlichte Implementierung mit denselben Zufallszahlen, Q-Tabellen exakt gleich; (5) der Dauerzyklus gegen
  eine Simulation der gierigen Strategien über viele Perioden.
- **Claims:** jede Zahl dieses READMEs in `tests/test_claims.py`, gelesen aus `akl_sweeps.json`; dazu eine Stichprobe, die vier Läufe neu rechnet.
- **Oberfläche:** AppTest (jeder Button, jeder Regler am Minimum und Maximum, Permalink begrenzt und rastet ein, Erklärungen, kein Rechtsurteil) und Footer-Test.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `akl_algorithm.py` | Kern: Markt, Q-Lernen, Dauerzyklus, Abweichungstest |
| `akl_constants.py` | Modell- und Lernparameter, Reglergrenzen, Presets |
| `akl_evaluation.py` | Analyse, Verdikt, Zugriff auf die vorgerechnete Tabelle |
| `akl_make_sweeps.py` | rechnet `akl_sweeps.json` vor |
| `akl_presets.py` | Permalink (Begrenzen und Einrasten), Presets |
| `akl_sweeps.json` | vorgerechnete Läufe (13 Konfigurationen × 20 Seeds) |
| `akl_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest, scipy, numpy (nur Orakel) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Handrechnung, Orakel, Claims, Presets, AppTest, Footer |

## Bewusst nicht umgesetzt

- **Mehr als zwei Anbieter, andere Nachfrage, Neuronale-Netz-Lernende:** eigene Modelle, hier nicht gerechnet.
- **Unvollkommene Beobachtung (Calvano et al. 2021):** wäre die Verbindung zu green-porter-demo; eigenes Stück.
- **Rechtliche Würdigung:** liegt außerhalb der Demo.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit und Plotly; die Lernschleife nutzt nur die Standardbibliothek.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
