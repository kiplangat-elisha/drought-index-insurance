# Drought index-insurance engine (Vegetation Condition Index)

Turns satellite vegetation data into drought classes and insurance payouts for smallholder farmers. Based on the drought-risk engine methodology I developed at GeoNova Concepts for index-based crop insurance (Makueni County).

## Why it matters

Drought is Kenya's costliest climate hazard for farmers. Index insurance pays out automatically when a satellite index shows drought, so farmers get money fast without field loss assessments. The trigger design decides who gets paid, so it has to be transparent and testable.

## Method

1. **VCI** (Kogan, 1995) compares current NDVI with the historical minimum and maximum for the same period of the year: `VCI = 100 × (NDVI − min) / (max − min)`.
2. **Drought classes** follow Kenya's National Drought Management Authority (NDMA) bands: extreme (<10), severe (10–20), moderate (20–35), normal (35–50), above normal (≥50).
3. **Payout** rises linearly from 0% at the trigger (VCI 35) to 100% at the exit (VCI 10), using the seasonal mean VCI per ward.

Handles cloud gaps (NaN), periods with no NDVI variability, and image stacks shaped (time, rows, cols).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                # run the tests
```

```bash
PYTHONPATH=src python -m drought_index.cli
```

Example output (synthetic 10-year series, short-rains season):

```
  unit  seasonal_vci  payout_share  sum_insured_kes  payout_kes   drought_class
Ward C          2.60          1.00     1,000,000.00 1,000,000.00 extreme drought
Ward B         17.67          0.69     1,000,000.00   693,268.00  severe drought
Ward A         93.59          0.00     1,000,000.00         0.00    above normal
```

## Using real data

Export monthly Sentinel-2 or MODIS NDVI per ward (e.g. from Google Earth Engine), pass the series to `vci_series`, and feed seasonal values to `season_payouts`. Trigger and exit levels should be calibrated against historical yield or loss data before any real insurance use.

## Author

Kiplangat Elisha · Climate adaptation and geospatial analyst · [geonova.co.ke](https://www.geonova.co.ke)
