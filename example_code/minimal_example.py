"""
Beispiel: Kompletter Workflow für eine Beobachtungsnacht
"""

from photometry_calibration import (
    PhotometryCalibrator, CoefficientMode, ExtinctionOrder,
    TransformationCoefficients, quick_calibrate
)
from astropy.table import Table
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
import astropy.units as u
import numpy as np

# =============================================================================
# BEISPIEL 1: Standard-Workflow mit PER_NIGHT Kalibration
# =============================================================================

print("=" * 70)
print("BEISPIEL 1: Per-Night Kalibration")
print("=" * 70)

# Simuliere Beobachtungsdaten (normalerweise aus deiner Pipeline)
np.random.seed(42)

def create_mock_frame(n_stars=50, airmass=1.3):
    """Erstellt simulierte Photometrie-Daten."""
    return Table({
        'star_id': [f"star_{i:03d}" for i in range(n_stars)],
        'ra': 180 + np.random.uniform(-0.2, 0.2, n_stars),
        'dec': 45 + np.random.uniform(-0.2, 0.2, n_stars),
        'mag_B': 12 + np.random.uniform(0, 3, n_stars) + np.random.normal(0, 0.02, n_stars),
        'mag_V': 11.5 + np.random.uniform(0, 3, n_stars) + np.random.normal(0, 0.02, n_stars),
        'mag_R': 11 + np.random.uniform(0, 3, n_stars) + np.random.normal(0, 0.02, n_stars),
        'err_B': np.full(n_stars, 0.015),
        'err_V': np.full(n_stars, 0.012),
        'err_R': np.full(n_stars, 0.010),
        'airmass': np.full(n_stars, airmass),
    })

# Mehrere Frames einer Nacht
frames = {
    'frame_001': create_mock_frame(airmass=1.1),
    'frame_002': create_mock_frame(airmass=1.2),
    'frame_003': create_mock_frame(airmass=1.4),
    'frame_004': create_mock_frame(airmass=1.6),
    'frame_005': create_mock_frame(airmass=1.3),
}

# Observatorium (für Airmass-Berechnung, falls nötig)
observatory = EarthLocation(lat=48.0*u.deg, lon=11.0*u.deg, height=500*u.m)

# Feldzentrum für APASS-Query
field_center = SkyCoord(ra=180*u.deg, dec=45*u.deg)

# Kalibrator erstellen
calibrator = PhotometryCalibrator(
    mode=CoefficientMode.PER_NIGHT,  # Koeffizienten über die Nacht mitteln
    extinction_order=ExtinctionOrder.FIRST,
    observatory_location=observatory
)

# APASS-Daten laden (in echtem Fall - hier mocken wir)
# calibrator.setup_apass(field_center, radius=20*u.arcmin)

# Für Demo: APASS-Daten simulieren
mock_apass = Table({
    'ra': 180 + np.random.uniform(-0.2, 0.2, 30),
    'dec': 45 + np.random.uniform(-0.2, 0.2, 30),
    'apass_id': [f"APASS_{i:06d}" for i in range(30)],
    'mag_std_B': 12 + np.random.uniform(0, 3, 30),
    'mag_std_V': 11.5 + np.random.uniform(0, 3, 30),
    'mag_std_R': 11 + np.random.uniform(0, 3, 30),
    'err_std_B': np.full(30, 0.02),
    'err_std_V': np.full(30, 0.02),
    'err_std_R': np.full(30, 0.02),
})
calibrator.apass_data = mock_apass

# Frames hinzufügen
for frame_id, data in frames.items():
    calibrator.add_frame(frame_id, data)

# Kalibration durchführen
results = calibrator.calibrate(
    filters=['B', 'V', 'R'],
    determine_color_terms=True,
    min_comparisons=3
)

print(calibrator.summary())

# Kalibrierte Photometrie abrufen
calibrated = calibrator.get_calibrated_photometry()
print(f"\nKalibrierte Tabelle: {len(calibrated)} Einträge")
print(calibrated.colnames)


# =============================================================================
# BEISPIEL 2: Per-Image Kalibration (für variable Bedingungen)
# =============================================================================

print("\n" + "=" * 70)
print("BEISPIEL 2: Per-Image Kalibration")
print("=" * 70)

calibrator_per_image = PhotometryCalibrator(
    mode=CoefficientMode.PER_IMAGE,  # Jedes Bild einzeln
    extinction_order=ExtinctionOrder.FIRST
)

calibrator_per_image.apass_data = mock_apass

for frame_id, data in frames.items():
    calibrator_per_image.add_frame(frame_id, data)

results_per_image = calibrator_per_image.calibrate(
    filters=['B', 'V', 'R'],
    determine_color_terms=False,  # Nur Nullpunkte, Color Terms festhalten
    min_comparisons=3
)

# Zeige Nullpunkt-Variation
print("\nNullpunkt-Variation über die Nacht:")
for frame_id, cal in results_per_image.items():
    if 'V' in cal.transformation:
        zp = cal.transformation['V'].zero_point
        print(f"  {frame_id}: ZP_V = {zp:.4f}")


# =============================================================================
# BEISPIEL 3: Feste Koeffizienten verwenden
# =============================================================================

print("\n" + "=" * 70)
print("BEISPIEL 3: Feste Koeffizienten (z.B. aus Literatur)")
print("=" * 70)

calibrator_fixed = PhotometryCalibrator(
    mode=CoefficientMode.FIXED,
    extinction_order=ExtinctionOrder.FIRST
)

# Koeffizienten manuell setzen (z.B. von früherem Run oder Literatur)
fixed_coeffs = {
    'B': TransformationCoefficients(
        filter_name='B',
        color_term=-0.05,
        color_term_err=0.01,
        zero_point=25.0,
        zero_point_err=0.02,
        color_index_filters=('B', 'V')
    ),
    'V': TransformationCoefficients(
        filter_name='V',
        color_term=0.03,
        color_term_err=0.01,
        zero_point=24.8,
        zero_point_err=0.02,
        color_index_filters=('B', 'V')
    ),
    'R': TransformationCoefficients(
        filter_name='R',
        color_term=-0.02,
        color_term_err=0.01,
        zero_point=24.5,
        zero_point_err=0.02,
        color_index_filters=('V', 'R')
    ),
}

calibrator_fixed.set_fixed_coefficients(fixed_coeffs)

for frame_id, data in frames.items():
    calibrator_fixed.add_frame(frame_id, data)

calibrator_fixed.calibrate(filters=['B', 'V', 'R'])
calibrated_fixed = calibrator_fixed.get_calibrated_photometry()

print(f"Mit festen Koeffizienten: {len(calibrated_fixed)} Einträge")


# =============================================================================
# BEISPIEL 4: Quick-Funktion für einfache Fälle
# =============================================================================

print("\n" + "=" * 70)
print("BEISPIEL 4: Quick-Calibrate Funktion")
print("=" * 70)

# Einfachster Aufruf (nur wenn APASS online verfügbar)
# result = quick_calibrate(
#     photometry_tables=frames,
#     field_center=field_center,
#     filters=['B', 'V', 'R'],
#     mode=CoefficientMode.PER_NIGHT
# )


# =============================================================================
# BEISPIEL 5: Kalibration speichern und laden
# =============================================================================

print("\n" + "=" * 70)
print("BEISPIEL 5: Kalibration speichern/laden")
print("=" * 70)

# Speichern
calibrator.save_calibration("calibration_night1.json")

# Laden in neuem Calibrator
new_calibrator = PhotometryCalibrator(mode=CoefficientMode.PER_NIGHT)
new_calibrator.load_calibration("calibration_night1.json")

# Kann jetzt auf neue Daten anwenden
print("Kalibration erfolgreich gespeichert und geladen!")


# =============================================================================
# BEISPIEL 6: Eigene Vergleichsstern-Selektion
# =============================================================================

print("\n" + "=" * 70)
print("BEISPIEL 6: Custom Comparison Star Selection")
print("=" * 70)

def my_comparison_selector(table):
    """
    Wähle nur Vergleichssterne die:
    - APASS-Match haben
    - V zwischen 11 und 14 mag
    - Keine Sättigung (err > 0.005)
    """
    mask = np.ones(len(table), dtype=bool)
    
    # APASS-Match
    if 'mag_std_V' in table.colnames:
        mask &= np.isfinite(table['mag_std_V'])
    
    # Magnitude-Bereich
    if 'mag_std_V' in table.colnames:
        mask &= (table['mag_std_V'] > 11) & (table['mag_std_V'] < 14)
    
    # Nicht saturiert
    if 'err_V' in table.colnames:
        mask &= table['err_V'] > 0.005
    
    return mask

calibrator_custom = PhotometryCalibrator(mode=CoefficientMode.PER_NIGHT)
calibrator_custom.apass_data = mock_apass

for frame_id, data in frames.items():
    calibrator_custom.add_frame(frame_id, data)

results_custom = calibrator_custom.calibrate(
    filters=['B', 'V', 'R'],
    comparison_selector=my_comparison_selector,
    min_comparisons=3
)

print(calibrator_custom.summary())
