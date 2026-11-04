import unittest
import cv2

import numpy as np

from sem_metrology.calibration import Calibration, detect_footer, detect_scale_bar, read_sem_companion_metadata
from sem_metrology.measurements import measure_cross_section, measure_layer_stack, measure_lines, measure_top_view
from sem_metrology.shapes import analyze_circular_outlines, analyze_rectangular_features, analyze_shapes, infer_lattice_periods, population_summary
from sem_metrology.lines import analyze_line_space, estimate_line_direction
from sem_metrology.models import Measurement
from sem_metrology.reporting import measurement_statistics
from sem_metrology.roughness import low_pass_three_point, spatial_descriptors
from sem_metrology.waveguide import analyze_waveguide
from sem_metrology.populations import infer_feature_classes
from sem_metrology.studies import common_study_metrics, dose_response_summary
from sem_metrology.profile import measure_lateral_profile


class CalibrationTests(unittest.TestCase):
    def test_line_calibration(self):
        calibration = Calibration.from_line(1.0, (10, 20), (110, 20))
        self.assertAlmostEqual(calibration.microns_per_pixel, 0.01)

    def test_zeiss_like_footer_and_green_bar(self):
        image = np.full((120, 200, 3), 50, dtype=np.uint8)
        image[90:] = 245
        image[102, 10:61] = (0, 220, 0)
        footer = detect_footer(image)
        bar = detect_scale_bar(image, footer)
        self.assertEqual(footer, 90)
        self.assertIsNotNone(bar)
        self.assertEqual(bar["pixel_length"], 50)

    def test_white_tick_ended_footer_scale_bar(self):
        image=np.full((180,320,3),55,np.uint8)
        image[125:, :]=35
        image[155,190:291]=235
        for x in (190,240,290): image[150:161,x]=235
        bar=detect_scale_bar(image,180)
        self.assertIsNotNone(bar)
        self.assertEqual(bar["style"],"white_tick_bar")
        self.assertAlmostEqual(bar["pixel_length"],100,delta=4)

    def test_dark_footer_and_tick_centres_define_scale_length(self):
        image=np.full((260,400,3),95,np.uint8)
        image[200:]=28
        cv2.line(image,(150,230),(350,230),(235,235,235),4)
        cv2.line(image,(150,222),(150,238),(235,235,235),2)
        cv2.line(image,(350,222),(350,238),(235,235,235),2)
        footer=detect_footer(image); bar=detect_scale_bar(image,footer)
        self.assertEqual(footer,200)
        self.assertIsNotNone(bar)
        self.assertAlmostEqual(bar["pixel_length"],200,delta=2)


class MeasurementTests(unittest.TestCase):
    def test_line_profile_reports_lateral_chords_and_gaps(self):
        signal=np.full(220,55.0)
        signal[25:55]=180; signal[90:115]=170; signal[155:195]=190
        rows,diagnostics=measure_lateral_profile(signal,.005,20,"bright")
        self.assertEqual(len(rows),3)
        self.assertAlmostEqual(rows[0]["Lateral chord"],.15,delta=.025)
        self.assertAlmostEqual(rows[0]["Edge-to-edge gap"],.175,delta=.035)
        self.assertEqual(diagnostics["polarity"],"bright")

    def test_class_inference_finds_two_separated_diameter_populations(self):
        rng=np.random.default_rng(12)
        values=np.r_[rng.normal(0.100,.002,18),rng.normal(0.130,.002,15)]
        model=infer_feature_classes(values)
        self.assertEqual(model["class_count"],2)
        self.assertEqual(sorted(model["counts"].tolist()),[15,18])
        self.assertGreater(model["confidence"],.8)

    def test_class_inference_does_not_split_one_normal_population(self):
        rng=np.random.default_rng(21)
        model=infer_feature_classes(rng.normal(.110,.006,40))
        self.assertEqual(model["class_count"],1)
        self.assertIsNone(model["labels"])

    def test_process_study_common_metric_and_response(self):
        records=[]
        for dose,diameter in ((100,95),(150,105),(200,115)):
            records.append({"factor_value":dose,"metrics":{"Diameter — Class A":{"mean":diameter,"std":2.0,"n":12},"Pitch":{"mean":220,"std":3,"n":12}}})
        self.assertEqual(common_study_metrics(records),["Diameter — Class A","Pitch"])
        summary=dose_response_summary(records,"Diameter — Class A")
        self.assertAlmostEqual(summary["slope"],.2)
        self.assertAlmostEqual(summary["r_squared"],1.0)
        self.assertEqual(summary["n"],3)

    def test_statistics_always_reports_n(self):
        rows = measurement_statistics([Measurement("CD", 0.1, "µm", "test", 1.0)])
        self.assertEqual(rows[0]["N"], 1)
        self.assertIn("dispersion unavailable", rows[0]["Status"])

    def test_contact_candidate_statistics_are_never_marked_ok(self):
        rows=measurement_statistics([Measurement("Contact candidate — Equivalent diameter",value,"µm","visible segment",.5) for value in (.10,.12,.11)])
        self.assertIn("REVIEW",rows[0]["Status"])
        combined=measurement_statistics([Measurement("Combined visible segments — Equivalent diameter",value,"µm","mixed",.5) for value in (.10,.12,.11)])
        self.assertIn("combined",combined[0]["Status"])

    def test_line_space_cd_pitch_and_roughness(self):
        image = np.zeros((120, 240), dtype=np.uint8)
        image[:, 20:40] = 220
        image[:, 100:120] = 220
        image[:, 180:200] = 220
        rows, _, bands = analyze_line_space(image, 0.01, "bright", 0.1, 0.4)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(bands), 3)
        self.assertAlmostEqual(rows[0]["Mean CD (µm)"], 0.2, delta=0.02)
        self.assertAlmostEqual(rows[0]["Pitch to next (µm)"], 0.8, delta=0.02)
        self.assertAlmostEqual(rows[0]["Duty cycle (%)"], 25, delta=3)
        self.assertFalse(any("Length" in key for key in rows[0]))

    def test_horizontal_grating_has_cd_not_line_length(self):
        image = np.zeros((240,120),dtype=np.uint8)
        for y in (20,100,180): image[y:y+20,:] = 220
        rows,_,_ = analyze_line_space(image,.01,"bright",.1,.4,"horizontal")
        self.assertEqual(len(rows),3)
        self.assertAlmostEqual(rows[0]["Mean CD (µm)"],.2,delta=.02)
        self.assertEqual(rows[0]["Orientation"],"horizontal")
    def test_disc_shape_statistics(self):
        import cv2
        image = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(image, (60, 100), 20, 230, -1)
        cv2.circle(image, (140, 100), 20, 230, -1)
        features, _, contours = analyze_shapes(image, 0.01, min_area_px=50)
        self.assertEqual(len(features), 2)
        self.assertEqual(len(contours), 2)
        self.assertAlmostEqual(features[0]["Equivalent diameter (µm)"], 0.4, delta=0.03)
        summary = population_summary(features)
        diameter = next(row for row in summary if row["Metric"] == "Equivalent diameter (µm)")
        self.assertEqual(diameter["N"], 2)

    def test_touching_irregular_particles_are_split_and_flagged(self):
        import cv2
        image=np.full((180,260),55,np.uint8)
        for center in ((80,90),(115,90),(180,90)):
            cv2.circle(image,center,24,210,-1,cv2.LINE_AA)
        features,_,contours=analyze_shapes(image,.005,"bright",200,5000,True,48,.82)
        self.assertEqual(len(features),3)
        self.assertEqual(len(contours),3)
        self.assertGreaterEqual(sum(bool(row["Contact split"]) for row in features),2)

    def test_taught_irregular_watershed_does_not_leak_into_background(self):
        import cv2
        rng=np.random.default_rng(77)
        image=np.clip(rng.normal(75,9,(220,360)),0,255).astype(np.uint8)
        for center in ((70,110),(175,110),(280,110)):
            cv2.circle(image,center,27,145,-1,cv2.LINE_AA)
            cv2.circle(image,center,27,225,3,cv2.LINE_AA)
        template=image[76:144,36:104]
        features,_,contours=analyze_shapes(image,.005,"bright",300,5000,True,68,.82,template)
        self.assertEqual(len(features),3)
        self.assertEqual(len(contours),3)
        self.assertLess(max(row["Area (µm²)"]/.005**2 for row in features),4000)

    def test_broken_rings_are_one_object_each(self):
        import cv2
        image = np.full((220, 300), 70, dtype=np.uint8)
        for center in ((70, 110), (150, 110), (230, 110)):
            cv2.ellipse(image, center, (28, 28), 0, 15, 330, 220, 3)
        features, _, circles = analyze_circular_outlines(image, 0.005, 0.20, 0.40, 0.82, True)
        self.assertEqual(len(features), 3)
        self.assertEqual(len(circles), 3)
        self.assertIn("Circle edge roughness 3σ (µm)", features[0])

    def test_spatial_roughness_descriptors_find_wavelength(self):
        x=np.arange(256,dtype=float); trace=np.sin(2*np.pi*x/32)
        correlation,wavelength=spatial_descriptors(trace,0.01)
        self.assertTrue(np.isfinite(correlation))
        self.assertAlmostEqual(wavelength,0.32,delta=0.02)
        self.assertLess(np.std(low_pass_three_point((-1.0)**x)),np.std((-1.0)**x))

    def test_taught_reference_refits_each_radius_independently(self):
        import cv2
        image = np.full((180, 300), 70, dtype=np.uint8)
        for center, radius in zip(((60, 90), (150, 90), (240, 90)), (24, 27, 30)):
            cv2.circle(image, center, radius, 225, 3)
        features, _, _ = analyze_circular_outlines(
            image, 0.005, 0.18, 0.36, 0.82, True, (32, 62, 56, 56)
        )
        diameters = sorted(row["Diameter (µm)"] for row in features)
        self.assertEqual(len(diameters), 3)
        self.assertGreater(diameters[-1] - diameters[0], 0.045)

    def test_taught_circle_array_keeps_recall_with_broad_global_size_range(self):
        import cv2
        rng=np.random.default_rng(31)
        image=np.clip(rng.normal(145,10,(300,440)),0,255).astype(np.uint8)
        for row in range(5):
            for column in range(7):
                center=(40+column*60,35+row*55)
                cv2.circle(image,center,10,72,-1,cv2.LINE_AA)
                cv2.circle(image,center,11,112,1,cv2.LINE_AA)
        features,_,_=analyze_circular_outlines(image,.004,.010,1.000,.82,True,(28,23,24,24))
        self.assertGreaterEqual(len(features),32)

    def test_taught_rectangular_array_reports_dimensions_and_period(self):
        import cv2
        image = np.full((180, 320), 60, dtype=np.uint8)
        for y in (45, 125):
            for x in (55, 155, 255):
                cv2.rectangle(image, (x-30, y-15), (x+30, y+15), 210, 2)
        rows, _, period_model = analyze_rectangular_features(image, 0.005, (20, 25, 70, 40), 0.82, True)
        self.assertEqual(len(rows), 6)
        self.assertAlmostEqual(np.mean([r["Length (µm)"] for r in rows]), 0.30, delta=0.04)
        self.assertAlmostEqual(np.mean([r["Mean width (µm)"] for r in rows]), 0.15, delta=0.04)
        self.assertAlmostEqual(np.nanmean([r["Period X (µm)"] for r in rows]), 0.50, delta=0.04)
        self.assertGreater(period_model["period_x"]["confidence"], 0.8)

    def test_lattice_period_survives_missing_sites_and_reports_secondary_gap(self):
        points = np.array([(x,y) for y in (0,80) for x in (0,100,200,550,650)], dtype=float)
        model = infer_lattice_periods(points, 0, 20, 60)
        self.assertAlmostEqual(model["fundamental_px"], 100, delta=3)
        self.assertTrue(any(abs(value-350) < 5 for value in model["secondary_px"]))

    def test_reviewed_line_distance(self):
        values = measure_lines([{"label": "CD", "x1": 0, "y1": 0, "x2": 3, "y2": 4}], 0.01)
        self.assertEqual(values[0].name, "CD")
        self.assertAlmostEqual(values[0].value, 0.05)

    def test_top_view_pitch(self):
        image = np.zeros((120, 240), dtype=np.uint8)
        for x in range(10, 240, 20):
            image[:, x:x + 5] = 220
        measurements, _ = measure_top_view(image, 0.01)
        pitch_x = next(m for m in measurements if m.name == "Mean pitch X")
        self.assertAlmostEqual(pitch_x.value, 0.2, places=3)

    def test_cross_section_dimensions(self):
        image = np.zeros((100, 160), dtype=np.uint8)
        image[20:80, 40:120] = 220
        measurements, _ = measure_cross_section(image, 0.01)
        values = {m.name: m.value for m in measurements}
        self.assertAlmostEqual(values["Feature height"], 0.60)
        self.assertAlmostEqual(values["Middle width"], 0.80)

    def test_cross_section_height_stops_at_substrate_interface(self):
        image=np.zeros((180,180),np.uint8); image[25:125,70:110]=220; image[125:150,:]=220
        measurements,diagnostics=measure_cross_section(image,0.01); values={m.name:m.value for m in measurements}
        self.assertAlmostEqual(values["Feature height"],1.00,delta=.03)
        self.assertEqual(diagnostics["height_datum"],"feature top to local substrate interface")

    def test_cross_section_top_cap_is_not_mistaken_for_substrate(self):
        image=np.zeros((220,100),np.uint8); image[20:30,:]=220; image[20:170,42:58]=220; image[170:195,:]=220
        measurements,diagnostics=measure_cross_section(image,0.01); values={m.name:m.value for m in measurements}
        self.assertAlmostEqual(values["Feature height"],1.50,delta=.03)
        self.assertEqual(diagnostics["height_line"][1],20)
        self.assertEqual(diagnostics["height_line"][3],170)

    def test_layer_stack_reports_interface_spacing(self):
        image=np.full((160,120),40,np.uint8); image[30:70]=110; image[70:125]=200
        measurements,diagnostics=measure_layer_stack(image,.01)
        thicknesses=[m.value for m in measurements]
        self.assertTrue(any(abs(value-.40)<.04 for value in thicknesses))
        self.assertTrue(any(abs(value-.55)<.04 for value in thicknesses))

    def test_layer_stack_operator_review_recalculates_ids_and_total(self):
        image=np.full((160,120),40,np.uint8)
        measurements,diagnostics=measure_layer_stack(image,.01,[20,50,90,130])
        values={m.name:m.value for m in measurements}
        self.assertAlmostEqual(values["L1 thickness (I1–I2)"],.30)
        self.assertAlmostEqual(values["L2 thickness (I2–I3)"],.40)
        self.assertAlmostEqual(values["L3 thickness (I3–I4)"],.40)
        self.assertAlmostEqual(values["Total stack height"],1.10)
        reviewed,_=measure_layer_stack(image,.01,[20,90,130])
        reviewed_values={m.name:m.value for m in reviewed}
        self.assertAlmostEqual(reviewed_values["L1 thickness (I1–I2)"],.70)
        self.assertAlmostEqual(reviewed_values["Total stack height"],1.10)

    def test_layer_stack_rejects_nonpersistent_internal_texture(self):
        rng=np.random.default_rng(4)
        image=np.full((170,180),45,np.float32)
        image[35:85]=110; image[85:140]=195
        image[95:125,60:85]+=rng.normal(0,45,(30,25))
        measurements,diagnostics=measure_layer_stack(np.clip(image,0,255).astype(np.uint8),.01)
        interfaces=diagnostics["interface_rows"]
        self.assertTrue(any(abs(row-35)<5 for row in interfaces))
        self.assertTrue(any(abs(row-85)<5 for row in interfaces))
        self.assertFalse(any(98<row<122 for row in interfaces))

    def test_diagonal_line_direction(self):
        import cv2
        image=np.zeros((240,240),np.uint8)
        for offset in range(-200,300,30): cv2.line(image,(offset,239),(offset+239,0),220,5)
        direction,confidence=estimate_line_direction(image)
        self.assertAlmostEqual(direction,135,delta=4)
        self.assertGreater(confidence,.5)

    def test_curved_waveguide_reports_normal_width_and_path(self):
        import cv2
        image=np.zeros((220,260),np.uint8)
        points=np.array([[20,170],[70,170],[110,130],[150,80],[230,80]],np.int32)
        cv2.polylines(image,[points],False,220,20,cv2.LINE_AA)
        rows,diagnostics=analyze_waveguide(image,0.01,np.ones_like(image,np.uint8))
        self.assertGreater(len(rows),20)
        self.assertAlmostEqual(diagnostics["mean_width"],0.20,delta=0.06)
        self.assertGreater(diagnostics["path_length"],2.0)


if __name__ == "__main__":
    unittest.main()
