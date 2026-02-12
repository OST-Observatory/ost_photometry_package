"""
photometry_calibration.py
=========================

Umfassendes Modul für photometrische Kalibration:
- Atmosphärische Extinktionskorrektur
- Magnituden-Transformation ins Standardsystem
- Differentielle Photometrie mit APASS-Unterstützung
- Flexible Koeffizientenbestimmung (pro Bild / pro Nacht)

Autor: [Dein Name]
Version: 2.0
"""

import numpy as np
from astropy.table import Table, vstack, unique
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u
from scipy.optimize import least_squares, minimize
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union, Literal
from enum import Enum, auto
from abc import ABC, abstractmethod
import warnings
from pathlib import Path
import json


# =============================================================================
# ENUMS UND KONFIGURATION
# =============================================================================

class CoefficientMode(Enum):
    """Modus für die Koeffizientenbestimmung."""
    PER_IMAGE = auto()      # Jedes Bild einzeln
    PER_NIGHT = auto()      # Gemittelt über eine Nacht
    FIXED = auto()          # Feste, vorgegebene Werte
    ENSEMBLE = auto()       # Ensemble-Photometrie (alle Bilder gemeinsam)


class ExtinctionOrder(Enum):
    """Ordnung der Extinktionskorrektur."""
    NONE = 0
    FIRST = 1       # m_0 = m - k' * X
    SECOND = 2      # m_0 = m - k' * X - k'' * X * (color)


# =============================================================================
# DATENKLASSEN
# =============================================================================

@dataclass
class ExtinctionCoefficients:
    """Extinktionskoeffizienten für einen Filter."""
    filter_name: str
    k_prime: float              # Erster Ordnung [mag/airmass]
    k_prime_err: float = 0.0
    k_second: float = 0.0       # Zweiter Ordnung [mag/airmass/mag_color]
    k_second_err: float = 0.0
    color_filter_1: str = ""    # Für k'' benötigter Farbindex
    color_filter_2: str = ""
    valid: bool = True
    
    def __repr__(self):
        s = f"k'_{self.filter_name} = {self.k_prime:.4f}±{self.k_prime_err:.4f}"
        if self.k_second != 0:
            ci = f"({self.color_filter_1}-{self.color_filter_2})"
            s += f", k''_{self.filter_name} = {self.k_second:.4f}±{self.k_second_err:.4f} × {ci}"
        return s


@dataclass
class TransformationCoefficients:
    """Transformationskoeffizienten für einen Filter."""
    filter_name: str
    color_term: float           # T
    color_term_err: float = 0.0
    zero_point: float = 0.0     # ZP
    zero_point_err: float = 0.0
    color_index_filters: Tuple[str, str] = ("B", "V")
    n_stars_used: int = 0
    rms_residual: float = 0.0
    
    def __repr__(self):
        ci = f"({self.color_index_filters[0]}-{self.color_index_filters[1]})"
        return (f"{self.filter_name}: T={self.color_term:.4f}±{self.color_term_err:.4f}, "
                f"ZP={self.zero_point:.4f}±{self.zero_point_err:.4f}, CI={ci}")


@dataclass
class CalibrationResult:
    """Container für Kalibrationsergebnisse eines Bildes/einer Nacht."""
    identifier: str                 # Bild-ID oder Nacht-ID
    timestamp: Optional[Time] = None
    extinction: Dict[str, ExtinctionCoefficients] = field(default_factory=dict)
    transformation: Dict[str, TransformationCoefficients] = field(default_factory=dict)
    n_comparison_stars: int = 0
    quality_flag: str = "OK"
    notes: str = ""


@dataclass
class DifferentialResult:
    """Ergebnis der differentiellen Photometrie für ein Target."""
    target_id: str
    filter_name: str
    mag_instrumental: float
    mag_instrumental_err: float
    mag_differential: float         # Relativ zu Ensemble
    mag_differential_err: float
    mag_calibrated: float           # Im Standardsystem
    mag_calibrated_err: float
    n_comparisons_used: int
    ensemble_zp: float              # Nullpunkt des Ensembles
    ensemble_zp_err: float
    airmass: float = 1.0
    timestamp: Optional[Time] = None


# =============================================================================
# STANDARD-EXTINKTIONSKOEFFIZIENTEN
# =============================================================================

# Typische Werte für gute Standorte (können überschrieben werden)
DEFAULT_EXTINCTION = {
    'U': ExtinctionCoefficients('U', k_prime=0.55, k_prime_err=0.05,
                                 k_second=0.03, color_filter_1='U', color_filter_2='B'),
    'B': ExtinctionCoefficients('B', k_prime=0.30, k_prime_err=0.03,
                                 k_second=0.02, color_filter_1='B', color_filter_2='V'),
    'V': ExtinctionCoefficients('V', k_prime=0.18, k_prime_err=0.02,
                                 k_second=0.01, color_filter_1='B', color_filter_2='V'),
    'R': ExtinctionCoefficients('R', k_prime=0.12, k_prime_err=0.02,
                                 k_second=0.01, color_filter_1='V', color_filter_2='R'),
    'I': ExtinctionCoefficients('I', k_prime=0.07, k_prime_err=0.02,
                                 k_second=0.005, color_filter_1='R', color_filter_2='I'),
}


# =============================================================================
# EXTINKTIONSKORREKTUR
# =============================================================================

class ExtinctionCorrector:
    """
    Korrigiert atmosphärische Extinktion.
    
    Die Atmosphäre absorbiert Licht wellenlängenabhängig.
    Korrektur: m_0 = m_obs - k' × X - k'' × X × (Farbindex)
    
    wobei X = Airmass (Luftmasse)
    """
    
    def __init__(
        self,
        coefficients: Optional[Dict[str, ExtinctionCoefficients]] = None,
        order: ExtinctionOrder = ExtinctionOrder.FIRST
    ):
        """
        Parameters
        ----------
        coefficients : dict, optional
            Extinktionskoeffizienten pro Filter.
            Falls None, werden Default-Werte verwendet.
        order : ExtinctionOrder
            FIRST (nur k') oder SECOND (k' und k'')
        """
        self.coefficients = coefficients or DEFAULT_EXTINCTION.copy()
        self.order = order
    
    def correct(
        self,
        data: Table,
        airmass_col: str = "airmass",
        mag_col_prefix: str = "mag_",
        output_prefix: str = "mag_ext_",
        filters: Optional[List[str]] = None,
        inplace: bool = False
    ) -> Table:
        """
        Wendet Extinktionskorrektur auf Magnituden an.
        
        Parameters
        ----------
        data : Table
            Eingabetabelle mit Magnituden und Airmass
        airmass_col : str
            Name der Airmass-Spalte
        mag_col_prefix : str
            Prefix der Magnituden-Spalten
        output_prefix : str
            Prefix für korrigierte Magnituden
        filters : list, optional
            Zu korrigierende Filter. Default: alle verfügbaren
        inplace : bool
            Wenn True, Original-Tabelle modifizieren
            
        Returns
        -------
        Table
            Tabelle mit extinktionskorrigierten Magnituden
        """
        if not inplace:
            data = data.copy()
        
        if airmass_col not in data.colnames:
            raise ValueError(f"Airmass-Spalte '{airmass_col}' nicht gefunden!")
        
        X = np.array(data[airmass_col])
        
        # Filter ermitteln
        if filters is None:
            filters = [col.replace(mag_col_prefix, "") 
                      for col in data.colnames if col.startswith(mag_col_prefix)]
        
        for filt in filters:
            mag_col = f"{mag_col_prefix}{filt}"
            if mag_col not in data.colnames:
                continue
            
            coeff = self.coefficients.get(filt)
            if coeff is None or not coeff.valid:
                warnings.warn(f"Keine Extinktionskoeffizienten für {filt}. Überspringe.")
                continue
            
            m_obs = np.array(data[mag_col], dtype=float)
            
            # Erste Ordnung
            correction = coeff.k_prime * X
            
            # Zweite Ordnung (falls aktiviert und Farbindex verfügbar)
            if self.order == ExtinctionOrder.SECOND and coeff.k_second != 0:
                ci_col1 = f"{mag_col_prefix}{coeff.color_filter_1}"
                ci_col2 = f"{mag_col_prefix}{coeff.color_filter_2}"
                
                if ci_col1 in data.colnames and ci_col2 in data.colnames:
                    color = np.array(data[ci_col1]) - np.array(data[ci_col2])
                    correction += coeff.k_second * X * color
            
            m_corrected = m_obs - correction
            data[f"{output_prefix}{filt}"] = m_corrected
        
        return data
    
    def determine_coefficients(
        self,
        observations: Table,
        filters: List[str],
        mag_col_prefix: str = "mag_",
        airmass_col: str = "airmass",
        star_id_col: str = "star_id",
        order: ExtinctionOrder = ExtinctionOrder.FIRST,
        sigma_clip: float = 3.0
    ) -> Dict[str, ExtinctionCoefficients]:
        """
        Bestimmt Extinktionskoeffizienten aus Beobachtungen bei
        verschiedenen Luftmassen (Bouguer-Methode).
        
        Benötigt Beobachtungen desselben Sterns bei verschiedenen
        Airmass-Werten (z.B. während des Aufgangs/Untergangs).
        
        Parameters
        ----------
        observations : Table
            Beobachtungen mit verschiedenen Airmass-Werten
        filters : list
            Zu bestimmende Filter
        star_id_col : str
            Spalte mit Stern-ID (für Gruppierung)
        order : ExtinctionOrder
            Ordnung der Extinktion
            
        Returns
        -------
        dict
            Extinktionskoeffizienten pro Filter
        """
        results = {}
        
        for filt in filters:
            mag_col = f"{mag_col_prefix}{filt}"
            if mag_col not in observations.colnames:
                continue
            
            # Gruppiere nach Stern
            unique_stars = np.unique(observations[star_id_col])
            
            all_X = []
            all_m = []
            all_star_idx = []
            
            for i, star in enumerate(unique_stars):
                mask = observations[star_id_col] == star
                X = np.array(observations[airmass_col][mask])
                m = np.array(observations[mag_col][mask])
                
                valid = np.isfinite(X) & np.isfinite(m)
                all_X.extend(X[valid])
                all_m.extend(m[valid])
                all_star_idx.extend([i] * np.sum(valid))
            
            all_X = np.array(all_X)
            all_m = np.array(all_m)
            all_star_idx = np.array(all_star_idx)
            
            # Fit: m = m_0 + k' * X (für jeden Stern eigenes m_0)
            # Matrix-Form für simultanen Fit
            n_stars = len(unique_stars)
            n_obs = len(all_X)
            
            # Design-Matrix: [1 für jeden Stern | X]
            A = np.zeros((n_obs, n_stars + 1))
            for i, star_idx in enumerate(all_star_idx):
                A[i, star_idx] = 1  # m_0 für diesen Stern
            A[:, -1] = all_X  # Airmass-Term
            
            # Least squares
            result, residuals, rank, s = np.linalg.lstsq(A, all_m, rcond=None)
            
            k_prime = result[-1]
            m_0s = result[:-1]
            
            # Fehlerabschätzung
            if len(all_m) > n_stars + 1:
                residual_m = all_m - A @ result
                rms = np.sqrt(np.sum(residual_m**2) / (len(all_m) - n_stars - 1))
                # Vereinfachte Fehlerabschätzung
                k_prime_err = rms / np.sqrt(np.sum((all_X - np.mean(all_X))**2))
            else:
                k_prime_err = 0.0
            
            results[filt] = ExtinctionCoefficients(
                filter_name=filt,
                k_prime=k_prime,
                k_prime_err=k_prime_err
            )
        
        self.coefficients.update(results)
        return results
    
    @staticmethod
    def calculate_airmass(
        coords: SkyCoord,
        obstime: Time,
        location: EarthLocation,
        method: str = "secz"
    ) -> np.ndarray:
        """
        Berechnet Airmass für gegebene Koordinaten und Zeit.
        
        Parameters
        ----------
        coords : SkyCoord
            Himmelskoordinaten
        obstime : Time
            Beobachtungszeit(en)
        location : EarthLocation
            Observatoriums-Position
        method : str
            'secz' (einfach) oder 'pickering' (genauer bei großem Zenitwinkel)
            
        Returns
        -------
        ndarray
            Airmass-Werte
        """
        altaz = coords.transform_to(AltAz(obstime=obstime, location=location))
        alt = altaz.alt.deg
        
        if method == "secz":
            # Einfache Näherung: X = sec(z) = 1/cos(z)
            zenith_angle = 90 - alt
            airmass = 1 / np.cos(np.radians(zenith_angle))
        elif method == "pickering":
            # Pickering (2002) - besser für große Zenitwinkel
            airmass = 1 / np.sin(np.radians(alt + 244 / (165 + 47 * alt**1.1)))
        else:
            raise ValueError(f"Unbekannte Methode: {method}")
        
        return np.clip(airmass, 1.0, 10.0)  # Sinnvolle Grenzen


# =============================================================================
# APASS KATALOG-INTERFACE
# =============================================================================

class APASSCatalog:
    """
    Interface zum APASS-Katalog für Vergleichssterne.
    
    APASS liefert B, V, g', r', i' Magnituden.
    Konversion zu R, I über empirische Relationen möglich.
    """
    
    # Typische Transformationen APASS -> Johnson-Cousins
    # Basierend auf verschiedenen Studien
    APASS_TRANSFORMS = {
        # R aus r' und i': R = r' - 0.2936*(r'-i') - 0.1439
        # I aus r' und i': I = r' - 1.2444*(r'-i') - 0.3820
        'R_from_ri': lambda r, i: r - 0.2936 * (r - i) - 0.1439,
        'I_from_ri': lambda r, i: r - 1.2444 * (r - i) - 0.3820,
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Parameters
        ----------
        cache_dir : str, optional
            Verzeichnis für gecachte Katalogdaten
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def query_region(
        self,
        center: SkyCoord,
        radius: u.Quantity,
        mag_limit: float = 16.0
    ) -> Table:
        """
        Fragt APASS-Daten für eine Himmelsregion ab.
        
        Parameters
        ----------
        center : SkyCoord
            Zentrum der Suchanfrage
        radius : Quantity
            Suchradius
        mag_limit : float
            Maximale V-Magnitude
            
        Returns
        -------
        Table
            Tabelle mit APASS-Sternen und Standard-Magnituden
        """
        from astroquery.vizier import Vizier
        
        # APASS DR9 Katalog
        v = Vizier(columns=['RAJ2000', 'DEJ2000', 'Bmag', 'Vmag', 
                           "g'mag", "r'mag", "i'mag",
                           'e_Bmag', 'e_Vmag', "e_g'mag", "e_r'mag", "e_i'mag"],
                   row_limit=-1)
        
        result = v.query_region(center, radius=radius, catalog='II/336/apass9')
        
        if not result:
            warnings.warn("Keine APASS-Daten gefunden!")
            return Table()
        
        apass = result[0]
        
        # Magnitude-Limit anwenden
        if 'Vmag' in apass.colnames:
            apass = apass[apass['Vmag'] < mag_limit]
        
        # Standardisierte Tabelle erstellen
        output = Table()
        output['ra'] = apass['RAJ2000']
        output['dec'] = apass['DEJ2000']
        output['apass_id'] = [f"APASS_{i:06d}" for i in range(len(apass))]
        
        # Direkt verfügbare Magnituden
        if 'Bmag' in apass.colnames:
            output['mag_std_B'] = apass['Bmag']
            output['err_std_B'] = apass['e_Bmag'] if 'e_Bmag' in apass.colnames else 0.02
        
        if 'Vmag' in apass.colnames:
            output['mag_std_V'] = apass['Vmag']
            output['err_std_V'] = apass['e_Vmag'] if 'e_Vmag' in apass.colnames else 0.02
        
        # R und I aus r' und i' transformieren
        if "r'mag" in apass.colnames and "i'mag" in apass.colnames:
            r_sloan = np.array(apass["r'mag"])
            i_sloan = np.array(apass["i'mag"])
            
            valid = np.isfinite(r_sloan) & np.isfinite(i_sloan)
            
            R_johnson = np.full(len(apass), np.nan)
            I_johnson = np.full(len(apass), np.nan)
            
            R_johnson[valid] = self.APASS_TRANSFORMS['R_from_ri'](r_sloan[valid], i_sloan[valid])
            I_johnson[valid] = self.APASS_TRANSFORMS['I_from_ri'](r_sloan[valid], i_sloan[valid])
            
            output['mag_std_R'] = R_johnson
            output['mag_std_I'] = I_johnson
            
            # Fehler abschätzen (inkl. Transformationsunsicherheit ~0.03 mag)
            output['err_std_R'] = np.sqrt(0.02**2 + 0.03**2)
            output['err_std_I'] = np.sqrt(0.02**2 + 0.03**2)
        
        # Sloan-Magnituden auch behalten
        for band in ['g', 'r', 'i']:
            col = f"{band}'mag"
            if col in apass.colnames:
                output[f'mag_std_{band}'] = apass[col]
        
        return output
    
    def crossmatch(
        self,
        sources: Table,
        apass_data: Table,
        ra_col: str = 'ra',
        dec_col: str = 'dec',
        match_radius: u.Quantity = 2.0 * u.arcsec
    ) -> Table:
        """
        Crossmatch zwischen gemessenen Quellen und APASS-Katalog.
        
        Parameters
        ----------
        sources : Table
            Tabelle mit gemessenen Quellen
        apass_data : Table
            APASS-Katalogdaten
        ra_col, dec_col : str
            Koordinaten-Spalten in sources
        match_radius : Quantity
            Maximaler Abstand für Match
            
        Returns
        -------
        Table
            Gematchte Tabelle mit instrumentellen UND Standard-Magnituden
        """
        from astropy.coordinates import match_coordinates_sky
        
        # Koordinaten erstellen
        source_coords = SkyCoord(sources[ra_col], sources[dec_col], unit='deg')
        apass_coords = SkyCoord(apass_data['ra'], apass_data['dec'], unit='deg')
        
        # Matching
        idx, sep, _ = match_coordinates_sky(source_coords, apass_coords)
        
        # Nur gute Matches behalten
        good_match = sep < match_radius
        
        # Ergebnis-Tabelle
        result = sources[good_match].copy()
        
        # APASS-Daten hinzufügen
        matched_apass = apass_data[idx[good_match]]
        for col in matched_apass.colnames:
            if col not in result.colnames:
                result[col] = matched_apass[col]
        
        result['match_sep_arcsec'] = sep[good_match].arcsec
        
        print(f"Crossmatch: {np.sum(good_match)}/{len(sources)} Quellen mit APASS gematcht")
        
        return result


# =============================================================================
# DIFFERENTIELLER PHOTOMETRIE-TRANSFORMER
# =============================================================================

class DifferentialPhotometer:
    """
    Führt differentielle Ensemble-Photometrie durch.
    
    Vorteile der differentiellen Photometrie:
    - Transparenzvariationen kürzen sich raus
    - Bessere Präzision als absolute Photometrie
    - Funktioniert auch bei nicht-photometrischen Bedingungen
    
    Workflow:
    1. Vergleichssterne mit APASS matchen
    2. Ensemble-Nullpunkt bestimmen
    3. Target-Magnituden transformieren
    """
    
    # Standard-Farbindizes
    DEFAULT_COLOR_INDICES = {
        'U': ('U', 'B'),
        'B': ('B', 'V'),
        'V': ('B', 'V'),
        'R': ('V', 'R'),
        'I': ('R', 'I'),
    }
    
    def __init__(
        self,
        color_indices: Optional[Dict] = None,
        extinction_corrector: Optional[ExtinctionCorrector] = None
    ):
        """
        Parameters
        ----------
        color_indices : dict, optional
            Farbindex-Zuordnungen pro Filter
        extinction_corrector : ExtinctionCorrector, optional
            Für Extinktionskorrektur vor der Transformation
        """
        self.color_indices = self.DEFAULT_COLOR_INDICES.copy()
        if color_indices:
            self.color_indices.update(color_indices)
        
        self.extinction = extinction_corrector
        self.calibrations: Dict[str, CalibrationResult] = {}
    
    def calibrate_frame(
        self,
        data: Table,
        frame_id: str,
        filters: List[str],
        comparison_mask: np.ndarray,
        mag_col_prefix: str = "mag_",
        std_col_prefix: str = "mag_std_",
        err_col_prefix: str = "err_",
        airmass_col: str = "airmass",
        sigma_clip: float = 2.5,
        min_comparisons: int = 3,
        determine_color_terms: bool = True
    ) -> CalibrationResult:
        """
        Kalibriert ein einzelnes Bild mit Ensemble-Photometrie.
        
        Parameters
        ----------
        data : Table
            Photometrie-Daten mit instrumentellen UND Standard-Magnituden
            (letztere aus APASS-Crossmatch)
        frame_id : str
            Identifikator für dieses Bild
        filters : list
            Zu kalibrierende Filter
        comparison_mask : ndarray
            Boolean-Maske für Vergleichssterne
        mag_col_prefix : str
            Prefix instrumenteller Magnituden
        std_col_prefix : str
            Prefix der Standard-Magnituden (aus APASS)
        sigma_clip : float
            Sigma für Ausreißer-Entfernung
        min_comparisons : int
            Mindestanzahl Vergleichssterne
        determine_color_terms : bool
            Wenn True, Color Terms aus den Daten bestimmen.
            Wenn False, nur Nullpunkt bestimmen (schneller, stabiler)
            
        Returns
        -------
        CalibrationResult
            Kalibrationsergebnis für dieses Bild
        """
        result = CalibrationResult(identifier=frame_id)
        
        # Extinktionskorrektur falls vorhanden
        if self.extinction is not None and airmass_col in data.colnames:
            data = self.extinction.correct(
                data, 
                airmass_col=airmass_col,
                mag_col_prefix=mag_col_prefix,
                output_prefix=mag_col_prefix,  # Überschreiben
                filters=filters,
                inplace=False
            )
        
        # Vergleichssterne selektieren
        comps = data[comparison_mask]
        
        for filt in filters:
            inst_col = f"{mag_col_prefix}{filt}"
            std_col = f"{std_col_prefix}{filt}"
            
            if inst_col not in comps.colnames or std_col not in comps.colnames:
                warnings.warn(f"Frame {frame_id}: Spalten für {filt} nicht vollständig")
                continue
            
            m_inst = np.array(comps[inst_col], dtype=float)
            m_std = np.array(comps[std_col], dtype=float)
            
            # Gültige Daten
            valid = np.isfinite(m_inst) & np.isfinite(m_std)
            
            if np.sum(valid) < min_comparisons:
                warnings.warn(f"Frame {frame_id}, Filter {filt}: "
                            f"Nur {np.sum(valid)} Vergleichssterne")
                continue
            
            # Farbindex berechnen
            ci_filters = self.color_indices.get(filt, ('B', 'V'))
            ci_inst_col1 = f"{mag_col_prefix}{ci_filters[0]}"
            ci_inst_col2 = f"{mag_col_prefix}{ci_filters[1]}"
            ci_std_col1 = f"{std_col_prefix}{ci_filters[0]}"
            ci_std_col2 = f"{std_col_prefix}{ci_filters[1]}"
            
            # Prüfen ob Farbindex verfügbar
            has_color = (ci_std_col1 in comps.colnames and 
                        ci_std_col2 in comps.colnames)
            
            if determine_color_terms and has_color:
                color_std = (np.array(comps[ci_std_col1]) - 
                           np.array(comps[ci_std_col2]))
                valid &= np.isfinite(color_std)
            else:
                color_std = np.zeros(len(comps))
            
            if np.sum(valid) < min_comparisons:
                continue
            
            # Iterativer Fit mit Sigma-Clipping
            mask = valid.copy()
            
            for iteration in range(5):
                n_before = np.sum(mask)
                
                delta_m = m_std[mask] - m_inst[mask]
                c = color_std[mask]
                
                if determine_color_terms and has_color and np.std(c) > 0.1:
                    # Fit mit Color Term
                    T, ZP, T_err, ZP_err = self._weighted_linear_fit(
                        c, delta_m, np.ones(np.sum(mask))
                    )
                else:
                    # Nur Nullpunkt
                    T, T_err = 0.0, 0.0
                    ZP = np.median(delta_m)
                    ZP_err = np.std(delta_m) / np.sqrt(np.sum(mask))
                
                # Residuen
                all_residuals = (m_std - m_inst) - (T * color_std + ZP)
                rms = np.nanstd(all_residuals[mask])
                
                # Sigma-Clipping
                new_mask = valid & (np.abs(all_residuals) < sigma_clip * rms)
                
                if np.sum(new_mask) == n_before or np.sum(new_mask) < min_comparisons:
                    break
                mask = new_mask
            
            result.transformation[filt] = TransformationCoefficients(
                filter_name=filt,
                color_term=T,
                color_term_err=T_err,
                zero_point=ZP,
                zero_point_err=ZP_err,
                color_index_filters=ci_filters,
                n_stars_used=int(np.sum(mask)),
                rms_residual=rms
            )
        
        result.n_comparison_stars = int(np.sum(comparison_mask))
        self.calibrations[frame_id] = result
        
        return result
    
    def calibrate_night(
        self,
        frames: Dict[str, Table],
        filters: List[str],
        comparison_mask_func,
        night_id: str = "night",
        **kwargs
    ) -> CalibrationResult:
        """
        Kalibriert eine ganze Nacht (gemittelte Koeffizienten).
        
        Parameters
        ----------
        frames : dict
            Dictionary {frame_id: Table} mit allen Frames der Nacht
        filters : list
            Zu kalibrierende Filter
        comparison_mask_func : callable
            Funktion die für jede Tabelle eine Maske zurückgibt:
            mask = comparison_mask_func(table)
        night_id : str
            Identifikator für diese Nacht
        **kwargs
            Weitere Argumente für calibrate_frame()
            
        Returns
        -------
        CalibrationResult
            Gemittelte Kalibration für die Nacht
        """
        frame_results = []
        
        for frame_id, data in frames.items():
            mask = comparison_mask_func(data)
            try:
                result = self.calibrate_frame(
                    data, frame_id, filters, mask, **kwargs
                )
                frame_results.append(result)
            except Exception as e:
                warnings.warn(f"Frame {frame_id} fehlgeschlagen: {e}")
        
        if not frame_results:
            raise ValueError("Keine Frames erfolgreich kalibriert!")
        
        # Koeffizienten mitteln
        night_result = CalibrationResult(identifier=night_id)
        
        for filt in filters:
            T_values = []
            ZP_values = []
            T_errs = []
            ZP_errs = []
            
            for fr in frame_results:
                if filt in fr.transformation:
                    tc = fr.transformation[filt]
                    T_values.append(tc.color_term)
                    ZP_values.append(tc.zero_point)
                    T_errs.append(tc.color_term_err)
                    ZP_errs.append(tc.zero_point_err)
            
            if not T_values:
                continue
            
            # Gewichtete Mittelung
            T_weights = 1 / (np.array(T_errs)**2 + 0.001**2)
            ZP_weights = 1 / (np.array(ZP_errs)**2 + 0.001**2)
            
            T_mean = np.average(T_values, weights=T_weights)
            ZP_mean = np.average(ZP_values, weights=ZP_weights)
            
            # Fehler aus Streuung (konservativer als formale Fehler)
            T_err = np.std(T_values) if len(T_values) > 1 else np.mean(T_errs)
            ZP_err = np.std(ZP_values) if len(ZP_values) > 1 else np.mean(ZP_errs)
            
            night_result.transformation[filt] = TransformationCoefficients(
                filter_name=filt,
                color_term=T_mean,
                color_term_err=T_err,
                zero_point=ZP_mean,
                zero_point_err=ZP_err,
                color_index_filters=frame_results[0].transformation[filt].color_index_filters,
                n_stars_used=sum(fr.transformation[filt].n_stars_used 
                               for fr in frame_results if filt in fr.transformation),
                rms_residual=np.mean([fr.transformation[filt].rms_residual 
                                     for fr in frame_results if filt in fr.transformation])
            )
        
        self.calibrations[night_id] = night_result
        return night_result
    
    def apply_calibration(
        self,
        data: Table,
        calibration: Union[str, CalibrationResult],
        filters: Optional[List[str]] = None,
        mag_col_prefix: str = "mag_",
        std_col_prefix: str = "mag_std_",
        output_prefix: str = "mag_cal_",
        err_col_prefix: str = "err_",
        output_err_prefix: str = "err_cal_",
        airmass_col: str = "airmass",
        max_iterations: int = 10,
        inplace: bool = False
    ) -> Table:
        """
        Wendet Kalibration auf Daten an.
        
        Parameters
        ----------
        data : Table
            Zu kalibrierende Daten
        calibration : str or CalibrationResult
            Kalibrations-ID oder CalibrationResult-Objekt
        filters : list, optional
            Zu transformierende Filter
        output_prefix : str
            Prefix für kalibrierte Magnituden
            
        Returns
        -------
        Table
            Tabelle mit kalibrierten Magnituden
        """
        if isinstance(calibration, str):
            if calibration not in self.calibrations:
                raise ValueError(f"Kalibration '{calibration}' nicht gefunden!")
            cal = self.calibrations[calibration]
        else:
            cal = calibration
        
        if not inplace:
            data = data.copy()
        
        # Extinktionskorrektur
        if self.extinction is not None and airmass_col in data.colnames:
            data = self.extinction.correct(
                data,
                airmass_col=airmass_col,
                mag_col_prefix=mag_col_prefix,
                output_prefix=mag_col_prefix,
                inplace=True
            )
        
        if filters is None:
            filters = list(cal.transformation.keys())
        
        # Iterative Transformation (wegen Farbindex im Standardsystem)
        std_mags = {}
        
        # Initialisierung mit instrumentellen Werten
        for filt in filters:
            col = f"{mag_col_prefix}{filt}"
            if col in data.colnames:
                std_mags[filt] = np.array(data[col], dtype=float)
        
        for iteration in range(max_iterations):
            max_change = 0.0
            
            for filt in filters:
                if filt not in cal.transformation:
                    continue
                
                tc = cal.transformation[filt]
                inst_col = f"{mag_col_prefix}{filt}"
                
                if inst_col not in data.colnames:
                    continue
                
                m_inst = np.array(data[inst_col], dtype=float)
                
                # Farbindex aus iterierten Standardmagnituden
                ci_f1, ci_f2 = tc.color_index_filters
                
                if ci_f1 in std_mags and ci_f2 in std_mags:
                    color = std_mags[ci_f1] - std_mags[ci_f2]
                else:
                    # Fallback: instrumentell
                    c1 = f"{mag_col_prefix}{ci_f1}"
                    c2 = f"{mag_col_prefix}{ci_f2}"
                    if c1 in data.colnames and c2 in data.colnames:
                        color = np.array(data[c1]) - np.array(data[c2])
                    else:
                        color = np.zeros(len(data))
                
                m_cal_new = m_inst + tc.color_term * color + tc.zero_point
                
                if filt in std_mags:
                    change = np.nanmax(np.abs(m_cal_new - std_mags[filt]))
                    max_change = max(max_change, change)
                
                std_mags[filt] = m_cal_new
            
            if max_change < 0.0001:
                break
        
        # Ergebnisse speichern
        for filt in filters:
            if filt in std_mags:
                data[f"{output_prefix}{filt}"] = std_mags[filt]
                
                # Fehlerfortpflanzung
                if filt in cal.transformation:
                    tc = cal.transformation[filt]
                    err_col = f"{err_col_prefix}{filt}"
                    
                    if err_col in data.colnames:
                        inst_err = np.array(data[err_col])
                        total_err = np.sqrt(inst_err**2 + tc.zero_point_err**2)
                        data[f"{output_err_prefix}{filt}"] = total_err
        
        return data
    
    @staticmethod
    def _weighted_linear_fit(x, y, weights):
        """Gewichteter linearer Fit."""
        W = np.sum(weights)
        Wx = np.sum(weights * x)
        Wy = np.sum(weights * y)
        Wxx = np.sum(weights * x**2)
        Wxy = np.sum(weights * x * y)
        
        denom = W * Wxx - Wx**2
        if abs(denom) < 1e-10:
            return 0.0, np.mean(y), 0.0, np.std(y)
        
        a = (W * Wxy - Wx * Wy) / denom
        b = (Wxx * Wy - Wx * Wxy) / denom
        
        residuals = y - (a * x + b)
        n = len(x)
        var = np.sum(weights * residuals**2) / (n - 2) if n > 2 else 0
        
        a_err = np.sqrt(var * W / denom) if denom > 0 else 0
        b_err = np.sqrt(var * Wxx / denom) if denom > 0 else 0
        
        return a, b, a_err, b_err


# =============================================================================
# HAUPTKLASSE FÜR DIE PIPELINE
# =============================================================================

class PhotometryCalibrator:
    """
    Hauptklasse für die Integration in die Pipeline.
    
    Kombiniert:
    - APASS-Katalogabfrage
    - Extinktionskorrektur
    - Differentielle Photometrie
    - Flexible Koeffizientenbestimmung
    
    Beispiel
    --------
    >>> calibrator = PhotometryCalibrator(mode=CoefficientMode.PER_NIGHT)
    >>> calibrator.setup_apass(field_center, radius=15*u.arcmin)
    >>> 
    >>> # Frames hinzufügen
    >>> for frame_id, photometry_table in frames.items():
    >>>     calibrator.add_frame(frame_id, photometry_table)
    >>> 
    >>> # Kalibrieren
    >>> results = calibrator.calibrate(filters=['B', 'V', 'R'])
    >>> 
    >>> # Ergebnisse abrufen
    >>> calibrated = calibrator.get_calibrated_photometry()
    """
    
    def __init__(
        self,
        mode: CoefficientMode = CoefficientMode.PER_NIGHT,
        extinction_order: ExtinctionOrder = ExtinctionOrder.FIRST,
        extinction_coefficients: Optional[Dict[str, ExtinctionCoefficients]] = None,
        observatory_location: Optional[EarthLocation] = None,
        color_indices: Optional[Dict] = None
    ):
        """
        Parameters
        ----------
        mode : CoefficientMode
            PER_IMAGE: Koeffizienten für jedes Bild einzeln
            PER_NIGHT: Gemittelte Koeffizienten pro Nacht
            FIXED: Vorgegebene Koeffizienten verwenden
            ENSEMBLE: Alle Bilder gemeinsam fitten
        extinction_order : ExtinctionOrder
            Ordnung der Extinktionskorrektur
        extinction_coefficients : dict, optional
            Eigene Extinktionskoeffizienten
        observatory_location : EarthLocation, optional
            Für Airmass-Berechnung
        color_indices : dict, optional
            Farbindex-Zuordnungen
        """
        self.mode = mode
        self.location = observatory_location
        
        # Extinktion
        self.extinction = ExtinctionCorrector(
            coefficients=extinction_coefficients,
            order=extinction_order
        )
        
        # APASS
        self.apass = APASSCatalog()
        self.apass_data: Optional[Table] = None
        
        # Photometer
        self.photometer = DifferentialPhotometer(
            color_indices=color_indices,
            extinction_corrector=self.extinction
        )
        
        # Daten-Speicher
        self.frames: Dict[str, Table] = {}
        self.frame_metadata: Dict[str, dict] = {}
        
        # Feste Koeffizienten (für FIXED-Modus)
        self.fixed_calibration: Optional[CalibrationResult] = None
    
    def setup_apass(
        self,
        center: SkyCoord,
        radius: u.Quantity = 15 * u.arcmin,
        mag_limit: float = 16.0
    ):
        """
        Lädt APASS-Katalog für das Beobachtungsfeld.
        
        Parameters
        ----------
        center : SkyCoord
            Feldzentrum
        radius : Quantity
            Feldradius
        mag_limit : float
            Magnitude-Limit für Vergleichssterne
        """
        print(f"Lade APASS-Daten für Feld bei {center.to_string('hmsdms')}...")
        self.apass_data = self.apass.query_region(center, radius, mag_limit)
        print(f"  → {len(self.apass_data)} APASS-Sterne geladen")
    
    def add_frame(
        self,
        frame_id: str,
        data: Table,
        obstime: Optional[Time] = None,
        airmass: Optional[float] = None,
        target_coords: Optional[SkyCoord] = None,
        ra_col: str = 'ra',
        dec_col: str = 'dec'
    ):
        """
        Fügt ein Bild zur Kalibration hinzu.
        
        Parameters
        ----------
        frame_id : str
            Eindeutiger Identifikator
        data : Table
            Photometrie-Daten (instrumentelle Magnituden)
        obstime : Time, optional
            Beobachtungszeit (für Airmass-Berechnung)
        airmass : float, optional
            Mittlere Airmass (falls nicht automatisch berechnen)
        target_coords : SkyCoord, optional
            Koordinaten des Targets (für Airmass)
        ra_col, dec_col : str
            Koordinaten-Spalten
        """
        data = data.copy()
        
        # Airmass berechnen oder hinzufügen
        if 'airmass' not in data.colnames:
            if airmass is not None:
                data['airmass'] = airmass
            elif obstime is not None and self.location is not None:
                # Berechne für jede Quelle
                coords = SkyCoord(data[ra_col], data[dec_col], unit='deg')
                data['airmass'] = self.extinction.calculate_airmass(
                    coords, obstime, self.location
                )
        
        # APASS-Crossmatch
        if self.apass_data is not None:
            data = self.apass.crossmatch(
                data, self.apass_data, ra_col=ra_col, dec_col=dec_col
            )
        
        self.frames[frame_id] = data
        self.frame_metadata[frame_id] = {
            'obstime': obstime,
            'airmass_mean': np.nanmean(data['airmass']) if 'airmass' in data.colnames else None
        }
        
        print(f"Frame '{frame_id}' hinzugefügt: {len(data)} Quellen")
    
    def set_fixed_coefficients(
        self,
        coefficients: Dict[str, TransformationCoefficients]
    ):
        """
        Setzt feste Koeffizienten für den FIXED-Modus.
        
        Parameters
        ----------
        coefficients : dict
            {filter_name: TransformationCoefficients}
        """
        self.fixed_calibration = CalibrationResult(
            identifier="fixed",
            transformation=coefficients
        )
        print("Feste Koeffizienten gesetzt:")
        for filt, tc in coefficients.items():
            print(f"  {tc}")
    
    def calibrate(
        self,
        filters: List[str],
        comparison_selector: Optional[callable] = None,
        determine_color_terms: bool = True,
        min_comparisons: int = 5,
        sigma_clip: float = 2.5
    ) -> Dict[str, CalibrationResult]:
        """
        Führt die Kalibration durch.
        
        Parameters
        ----------
        filters : list
            Zu kalibrierende Filter
        comparison_selector : callable, optional
            Funktion zur Auswahl von Vergleichssternen.
            Signatur: mask = selector(table)
            Default: Alle Sterne mit APASS-Match und guten Daten
        determine_color_terms : bool
            Wenn True, Color Terms aus Daten bestimmen
        min_comparisons : int
            Mindestanzahl Vergleichssterne
        sigma_clip : float
            Sigma für Ausreißer
            
        Returns
        -------
        dict
            Kalibrationsergebnisse pro Frame/Nacht
        """
        if not self.frames:
            raise ValueError("Keine Frames hinzugefügt!")
        
        # Default-Selektor: Alle mit APASS-Match
        if comparison_selector is None:
            def comparison_selector(table):
                mask = np.ones(len(table), dtype=bool)
                for filt in filters:
                    std_col = f"mag_std_{filt}"
                    if std_col in table.colnames:
                        mask &= np.isfinite(table[std_col])
                return mask
        
        results = {}
        
        if self.mode == CoefficientMode.FIXED:
            if self.fixed_calibration is None:
                raise ValueError("FIXED-Modus aber keine Koeffizienten gesetzt!")
            for frame_id in self.frames:
                results[frame_id] = self.fixed_calibration
        
        elif self.mode == CoefficientMode.PER_IMAGE:
            for frame_id, data in self.frames.items():
                mask = comparison_selector(data)
                result = self.photometer.calibrate_frame(
                    data, frame_id, filters, mask,
                    determine_color_terms=determine_color_terms,
                    min_comparisons=min_comparisons,
                    sigma_clip=sigma_clip
                )
                results[frame_id] = result
                print(f"Frame {frame_id}: {result.n_comparison_stars} Vergleichssterne")
        
        elif self.mode == CoefficientMode.PER_NIGHT:
            result = self.photometer.calibrate_night(
                self.frames, filters, comparison_selector,
                night_id="night_combined",
                determine_color_terms=determine_color_terms,
                min_comparisons=min_comparisons,
                sigma_clip=sigma_clip
            )
            # Gleiche Kalibration für alle Frames
            for frame_id in self.frames:
                results[frame_id] = result
            print(f"Nacht-Kalibration: {len(self.frames)} Frames kombiniert")
        
        elif self.mode == CoefficientMode.ENSEMBLE:
            # Alle Daten zusammen fitten
            combined = vstack(list(self.frames.values()))
            combined_mask = comparison_selector(combined)
            result = self.photometer.calibrate_frame(
                combined, "ensemble", filters, combined_mask,
                determine_color_terms=determine_color_terms,
                min_comparisons=min_comparisons,
                sigma_clip=sigma_clip
            )
            for frame_id in self.frames:
                results[frame_id] = result
        
        self._calibration_results = results
        return results
    
    def get_calibrated_photometry(
        self,
        output_prefix: str = "mag_cal_",
        target_selector: Optional[callable] = None
    ) -> Table:
        """
        Wendet Kalibration an und gibt kalibrierte Tabelle zurück.
        
        Parameters
        ----------
        output_prefix : str
            Prefix für kalibrierte Magnituden
        target_selector : callable, optional
            Funktion zur Target-Auswahl.
            Default: Alle Quellen
            
        Returns
        -------
        Table
            Kalibrierte Photometrie
        """
        if not hasattr(self, '_calibration_results'):
            raise ValueError("Erst calibrate() aufrufen!")
        
        all_results = []
        
        for frame_id, data in self.frames.items():
            cal = self._calibration_results[frame_id]
            
            calibrated = self.photometer.apply_calibration(
                data, cal,
                output_prefix=output_prefix,
                inplace=False
            )
            
            calibrated['frame_id'] = frame_id
            
            if target_selector is not None:
                mask = target_selector(calibrated)
                calibrated = calibrated[mask]
            
            all_results.append(calibrated)
        
        return vstack(all_results) if all_results else Table()
    
    def summary(self) -> str:
        """Gibt eine Zusammenfassung der Kalibration."""
        lines = ["=" * 70,
                 "PHOTOMETRY CALIBRATION SUMMARY",
                 "=" * 70,
                 f"Mode: {self.mode.name}",
                 f"Frames: {len(self.frames)}",
                 f"APASS Stars: {len(self.apass_data) if self.apass_data else 'not loaded'}",
                 "-" * 70]
        
        if hasattr(self, '_calibration_results'):
            # Zeige Koeffizienten
            example_cal = list(self._calibration_results.values())[0]
            lines.append("Transformation Coefficients:")
            for filt, tc in example_cal.transformation.items():
                lines.append(f"  {tc}")
                lines.append(f"    N={tc.n_stars_used}, RMS={tc.rms_residual:.4f} mag")
        
        lines.append("=" * 70)
        return "\n".join(lines)
    
    def save_calibration(self, filename: str):
        """Speichert Kalibrationsergebnisse."""
        if not hasattr(self, '_calibration_results'):
            raise ValueError("Keine Kalibration zum Speichern!")
        
        output = {
            'mode': self.mode.name,
            'extinction_order': self.extinction.order.name,
            'calibrations': {}
        }
        
        for cal_id, cal in self._calibration_results.items():
            output['calibrations'][cal_id] = {
                'transformation': {
                    filt: {
                        'color_term': tc.color_term,
                        'color_term_err': tc.color_term_err,
                        'zero_point': tc.zero_point,
                        'zero_point_err': tc.zero_point_err,
                        'color_index_filters': tc.color_index_filters,
                        'n_stars_used': tc.n_stars_used,
                        'rms_residual': tc.rms_residual
                    }
                    for filt, tc in cal.transformation.items()
                }
            }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Kalibration gespeichert: {filename}")
    
    def load_calibration(self, filename: str):
        """Lädt Kalibrationsergebnisse."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self._calibration_results = {}
        
        for cal_id, cal_data in data['calibrations'].items():
            transformation = {}
            for filt, tc_data in cal_data['transformation'].items():
                transformation[filt] = TransformationCoefficients(
                    filter_name=filt,
                    color_term=tc_data['color_term'],
                    color_term_err=tc_data['color_term_err'],
                    zero_point=tc_data['zero_point'],
                    zero_point_err=tc_data['zero_point_err'],
                    color_index_filters=tuple(tc_data['color_index_filters']),
                    n_stars_used=tc_data['n_stars_used'],
                    rms_residual=tc_data['rms_residual']
                )
            
            self._calibration_results[cal_id] = CalibrationResult(
                identifier=cal_id,
                transformation=transformation
            )
        
        print(f"Kalibration geladen: {filename}")


# =============================================================================
# CONVENIENCE-FUNKTIONEN
# =============================================================================

def quick_calibrate(
    photometry_tables: Dict[str, Table],
    field_center: SkyCoord,
    filters: List[str],
    mode: CoefficientMode = CoefficientMode.PER_NIGHT,
    field_radius: u.Quantity = 15 * u.arcmin,
    **kwargs
) -> Table:
    """
    Schnelle Kalibration für typische Anwendungsfälle.
    
    Parameters
    ----------
    photometry_tables : dict
        {frame_id: Table} mit instrumentellen Magnituden
    field_center : SkyCoord
        Feldzentrum für APASS-Query
    filters : list
        Zu kalibrierende Filter (z.B. ['B', 'V', 'R'])
    mode : CoefficientMode
        Kalibrations-Modus
    field_radius : Quantity
        Radius für APASS-Query
        
    Returns
    -------
    Table
        Kalibrierte Photometrie aller Frames
    """
    calibrator = PhotometryCalibrator(mode=mode)
    calibrator.setup_apass(field_center, field_radius)
    
    for frame_id, table in photometry_tables.items():
        calibrator.add_frame(frame_id, table)
    
    calibrator.calibrate(filters, **kwargs)
    
    print(calibrator.summary())
    
    return calibrator.get_calibrated_photometry()
