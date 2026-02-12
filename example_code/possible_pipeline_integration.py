"""
Beispiel: Integration in eine bestehende Pipeline
"""

# In deiner Pipeline-Klasse:
class MyPhotometryPipeline:
    def __init__(self, ...):
        # ... andere Initialisierung ...
        
        # Kalibrator als Attribut
        self.calibrator = None
    
    def setup_calibration(
        self,
        field_center: SkyCoord,
        mode: str = "per_night",  # oder "per_image", "fixed"
        **kwargs
    ):
        """Initialisiert die Kalibration."""
        mode_map = {
            "per_night": CoefficientMode.PER_NIGHT,
            "per_image": CoefficientMode.PER_IMAGE,
            "fixed": CoefficientMode.FIXED,
            "ensemble": CoefficientMode.ENSEMBLE,
        }
        
        self.calibrator = PhotometryCalibrator(
            mode=mode_map[mode],
            **kwargs
        )
        self.calibrator.setup_apass(field_center)
    
    def process_frame(self, fits_file, ...):
        """Verarbeitet ein einzelnes Bild."""
        # ... Apertur-Photometrie etc. ...
        photometry_table = self.do_photometry(fits_file)
        
        # Frame zum Kalibrator hinzufügen
        if self.calibrator is not None:
            obstime = Time(fits_file.header['DATE-OBS'])
            self.calibrator.add_frame(
                frame_id=fits_file.filename,
                data=photometry_table,
                obstime=obstime
            )
        
        return photometry_table
    
    def finalize_calibration(self, filters=['B', 'V', 'R']):
        """Führt Kalibration durch nach Verarbeitung aller Frames."""
        if self.calibrator is None:
            raise ValueError("Kalibration nicht initialisiert!")
        
        self.calibrator.calibrate(filters)
        return self.calibrator.get_calibrated_photometry()
