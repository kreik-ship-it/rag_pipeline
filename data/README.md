# Daten

Die PDFs sind urheberrechtlich geschützte Veröffentlichungen und werden **nicht** im
Repository versioniert (`data/*.pdf` ist gitignoriert). Für eine Reproduktion bitte
folgende öffentlich verfügbare Dokumente herunterladen und unter genau diesen
Dateinamen in `data/` ablegen:

| Datei | Dokument | Herausgeber |
|---|---|---|
| `MonitoringberichtEnergie2024.pdf` | Monitoringbericht 2024 (348 S.) | Bundesnetzagentur & Bundeskartellamt |
| `DE2024_ISE_Studie_Stromgestehungskosten_Erneuerbare_Energien.pdf` | Stromgestehungskosten Erneuerbare Energien, Juli 2024 (52 S.) | Fraunhofer ISE |
| `2021-02-17_cc_08-2021_transformation_stromerzeugung_2050_0.pdf` | Climate Change 08/2021: Strommarkt und Klimaschutz – Transformation der Stromerzeugung bis 2050 (254 S.) | Umweltbundesamt / Connect Energy Economics |

Danach:

```bash
energy-rag ingest     # baut den Vektorindex in data/vector_store/ (gitignoriert)
energy-rag annotate   # nur nötig, wenn das Testset geändert wurde
```

`eval/testset.jsonl` ist versioniert: 36 handannotierte Fragen (34 beantwortbar,
2 bewusst nicht beantwortbar) mit Referenzantwort, Evidenz-Mustern und den daraus
automatisch bestimmten Belegseiten.
