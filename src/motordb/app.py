import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from extract_curves_fitted import extract_curve, get_linear_torque_coeffs, get_interpolated_rpms, convert_torque_to_pressure, get_interpolated_value
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import os

import pint

import sys
import logging
from pathlib import Path

# Initialize unit registry
ureg = pint.UnitRegistry()

# Basic force units
ureg.define('poundforce = 4.448222*N = lbf = LBF')
ureg.define('kgforce = 9.80665*N = kgf = KGF = kilogramforce')

# Volume units
ureg.define('barrel = 42*gallon = bbl = BBL')
ureg.define('thousand_cubic_feet = 1000*feet**3 = MCF')
ureg.define('million_cubic_feet = 1000000*feet**3 = MMCF')

# Flow rate units
ureg.define('cubic_meter_per_second = meter**3 / second = cms = CMS')
ureg.define('cubic_feet_per_minute = feet**3 / min = cfm = CFM')
ureg.define('cubic_meter_per_minute = meter**3 / minute = cmm = CMM')
ureg.define('liter_per_second = liter / second = lps = LPS')
ureg.define('gallon_per_second = gallon / second = gps = GPS')
ureg.define('gallon_per_minute = gallon / minute = gpm = GPM')
ureg.define('liters_per_minute = liter / minute = lpm = LPM')
ureg.define('barrels_per_minute = barrel / minute = bpm = BPM')
ureg.define('barrels_per_hour = barrel / hour = bph = BPH')
ureg.define('barrels_per_day = barrel / day = bpd = BPD')
ureg.define('thousand_cubic_feet_per_day = MCF / day = mcfd = MCFD')

# Pressure units
ureg.define('pascal = kilogram / (meter * second**2)')
ureg.define('bar = 100000 * pascal')
ureg.define('atm = 101325 * pascal')
ureg.define('pound_per_foot_squared = poundforce / foot**2')
ureg.define('ksc = kilogramforce / centimetre**2 = KSC')
ureg.define('pound_per_square_inch = poundforce / inch**2 = psi = PSI')
ureg.define('kilopascal = 1000 * pascal = kPa = KPA')
ureg.define('megapascal = 1000000 * pascal = MPa = MPA')

# Torque units
ureg.define('newton_meter = newton * meter = Nm = NM')
ureg.define('foot_pound = foot * poundforce = ft_lb = FT_LB')
ureg.define('kgf_meter = kilogramforce * meter = kgf_m = KGF_M')
ureg.define('kilonewton_meter = 1000 * newton * meter = kNm = KNM')
ureg.define('kilojoule = 1000 * joule = kJ = KJ')

# Power units
ureg.define('horsepower = 745.7 * watt = hp = HP')
ureg.define('mechanical_horsepower = 745.7 * watt = mhp = MHP')
ureg.define('metric_horsepower = 735.49875 * watt = PS')
ureg.define('kilowatt = 1000 * watt = kW = KW')
ureg.define('megawatt = 1000000 * watt = MW')


# Set up directories
user_home = os.path.expanduser("~/Documents")
mud_motor_dir = os.path.join(user_home, "mud_motor")

# Create the mud_motor directory if it doesn't exist
os.makedirs(mud_motor_dir, exist_ok=True)
log_file = os.path.join(mud_motor_dir, "motordb_log.txt")

if os.path.isfile(log_file):
    try:
        os.remove(log_file)
        print(f"Previous log file deleted: {log_file}")
    except Exception as e:
        print(f"Error deleting previous log file: {e}")

logging.basicConfig(filename=str(log_file), level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger for all console output
console_logger = logging.getLogger('Console')

# Redirect stdout and stderr
class StreamToLogger:
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

sys.stdout = StreamToLogger(console_logger, logging.INFO)
sys.stderr = StreamToLogger(console_logger, logging.ERROR)

# Define paths for temp image and database file
temp_image_path = os.path.join(mud_motor_dir, "temp.png")
database_path = os.path.join(mud_motor_dir, "motor_db.json")


with open("./components/jsui/index.html", "r") as f:
    HTML_CONTENT = f.read()

class CustomHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode())
        else:
            super().do_GET()

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, CustomHandler)
    print(f"Serving on port {port}")
    httpd.serve_forever()

class CurveExtractorApp(toga.App):
    def startup(self):
        self.main_window = toga.MainWindow(title="Mud Motor Performance Estimation Tool")

        self.webview = toga.WebView(
            style=Pack(flex=1),
            on_webview_load=self.on_webview_load
        )
        
        # Add the Calculate button
        self.calculate_button = toga.Button(
            'Calculate',
            on_press=self.on_calculate,
            style=Pack(padding=10)
        )

        # Add the Convert button
        self.convert_button = toga.Button(
            'Convert',
            on_press=self.on_convert,
            style=Pack(padding=10),
            enabled=False
        )

        # Add Numeric Inputs for Torque and Flowrate
        self.torque_input = toga.NumberInput(
            style=Pack(padding=10),
        )
        self.flowrate_input = toga.NumberInput(
            style=Pack(padding=10),
        )

        # Label to display results
        self.result_label = toga.Label(
            'RPM',
            style=Pack(padding=10)
        )

        # Add the mudmotor name/identifier input
        self.mudmotor_id_input = toga.TextInput(
            placeholder="Mud Motor ID",
            style=Pack(padding=10, flex=1)
        )

        # Add the Save to Database button
        self.save_button = toga.Button(
            'Save to Database',
            on_press=self.on_save_to_database,
            style=Pack(padding=10),
            enabled=False
        )

        # Create a row for the Torque/Flowrate inputs and Convert button
        box2 = toga.Box(
            children=[
                toga.Label("Torque"),
                self.torque_input,
                toga.Label("Flow rate"),
                self.flowrate_input,
                self.result_label,
                self.convert_button
            ],
            style=Pack(direction=ROW)
        )

        # Create a row for the Mudmotor name/ID input and Save button
        box3 = toga.Box(
            children=[
                toga.Label("Mud Motor Name/ID"),
                self.mudmotor_id_input,
                self.save_button
            ],
            style=Pack(direction=ROW, padding_top=10)
        )

        # Main layout including the Calculate button, input boxes, and WebView
        box = toga.Box(
            children=[
                self.webview,
                self.calculate_button,
                box2,
                box3                
            ],
            style=Pack(direction=COLUMN)
        )

        self.main_window.content = box
        self.main_window.show()

        # Start the local server in a separate thread
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

        # Load the HTML content from the local server
        self.webview.url = 'http://localhost:8000'


    def on_webview_load(self, widget):
        print("loaded")
        #pass  # We don't need to do anything here for now

    async def on_calculate(self, widget):
        try:
            # Get the data stored in the WebView
            result = await self.webview.evaluate_javascript('getData()')
            print(result)
            data = json.loads(result)
            #print(data)
            lines = data['lines']
            point = data['point']
            image_base64 = data['imageBase64']
            image_data = base64.b64decode(image_base64.split(',')[1])
            image = Image.open(BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Save the image to a temporary file
            #temp_file_path = 'temp.png'
            #image.save(temp_image_path, format='PNG')
            pressure_units = data['pressureUnits']
            torque_units = data['torqueUnits']
            flowrate_units = data['flowRateUnits']
            rpm_units = data['rpmUnits']
            max_torque = float(lines['maxTorque'])
            min_torque = float(lines['minRPMTorque'])
            max_rpm = float(lines['maxRPM'])
            min_rpm = float(lines['minRPMTorque'])
            max_pressure = float(lines['maxPressure'])
            min_pressure = float(lines['minPressure'])
            
            data['maxFlowRateGPM'] = (float(data['maxFlowRateGPM']) * ureg.parse_expression(data['flowRateUnits'])).to(ureg.gallon / ureg.minute).magnitude
            data['minFlowRateGPM'] = (float(data['minFlowRateGPM']) * ureg.parse_expression(data['flowRateUnits'])).to(ureg.gallon / ureg.minute).magnitude
            data['maxPressurepsi'] = (float(data['maxPressurepsi']) * ureg.parse_expression(data['pressureUnits'])).to(ureg.psi).magnitude
            data['minPressurepsi'] = (float(data['minPressurepsi']) * ureg.parse_expression(data['pressureUnits'])).to(ureg.psi).magnitude
            data['maxTorqueftlb'] = (float(data['maxTorqueftlb']) * ureg.parse_expression(data['torqueUnits'])).to(ureg.foot * ureg.poundforce).magnitude
            data['minTorqueftlb'] = (float(data['minTorqueftlb']) * ureg.parse_expression(data['torqueUnits'])).to(ureg.foot * ureg.poundforce).magnitude
            
            tqscale = (float(data['maxTorqueftlb']) - float(data['minTorqueftlb'])) / (min_torque - max_torque)
            psiscale = (float(data['maxPressurepsi']) - float(data['minPressurepsi'])) / (max_pressure - min_pressure)
            
            torquepoint = [float(point['x']) - min_pressure, float(point['y']) - max_torque]
            torquepoint[1] = float(data['maxTorqueftlb']) - torquepoint[1] * tqscale
            torquepoint[0] = torquepoint[0] * psiscale
            
            print("picked torque point is: ", torquepoint[1], "ft.lb at ", torquepoint[0], "psi")
            
            ucColor = (data['upperCurveColor'])
            ucColor = [int(x) for x in ucColor[4:-1].split(', ')]
            lcColor = (data['lowerCurveColor'])
            lcColor = [int(x) for x in lcColor[4:-1].split(', ')]
            print(ucColor)
            # Assemble crop coordinates for torque and RPM
            crop_coords_torque = [(round(min_pressure), round(max_torque)), 
                                  (round(max_pressure), round(min_torque))]
            crop_coords_rpm = [(round(min_pressure), round(max_rpm)), 
                               (round(max_pressure), round(min_rpm))]
            lasso_points_upper = [(point['x'], point['y']) for point in data['brushPoints']['brush1']]
            lasso_points_lower = [(point['x'], point['y']) for point in data['brushPoints']['brush2']]

            print(crop_coords_rpm)
            print(crop_coords_torque)
            print(lasso_points_upper)
            print(lasso_points_lower)
            m, c, __ = get_linear_torque_coeffs(image_base64, float(data['maxTorqueftlb']), float(data['maxPressurepsi']), crop_coords_torque, torquepoint)

            
            coeffsUpper = extract_curve(
                img64=image_base64,
                crop_coords=crop_coords_rpm,
                x_min=float(data['minPressurepsi']),
                x_max=float(data['maxPressurepsi']),
                y_min=float(data['minRPMval']),
                y_max=float(data['maxRPMval']),
                color_value=ucColor,
                custom_x_values=[],
                lasso=lasso_points_upper
            )
            coeffsLower = extract_curve(
                img64=image_base64,
                crop_coords=crop_coords_rpm,
                x_min=float(data['minPressurepsi']),
                x_max=float(data['maxPressurepsi']),
                y_min=float(data['minRPMval']),
                y_max=float(data['maxRPMval']),
                color_value=lcColor,
                custom_x_values=[],
                lasso=lasso_points_lower
            )
            
            self.coeffs_list = [coeffsUpper, coeffsLower]
            self.em = m
            self.flow_list = [float(data['maxFlowRateGPM']), float(data['minFlowRateGPM'])]
            self.stallpressure = float(data['maxPressurepsi'])
            self.overspeed = float(data['maxRPMval'])
            self.torqueunits = data['torqueUnits']
            self.flowrateunits = data['flowRateUnits']
            # Enable the Convert button
            self.convert_button.enabled = True
            self.save_button.enabled = True
        except Exception as e:
            print("Something went wrong, Check values and try again")
            self.main_window.error_dialog('Error', f"Something went wrong, Check values and try again.\n\n{e}\n\nAre you sure all data is correctly entered?")
            
    async def on_convert(self, widget):
        try:
            giventorque = (float(self.torque_input.value) * ureg.parse_expression(self.torqueunits)).to(ureg.ft_lb).magnitude
            givenflow = (float(self.flowrate_input.value) * ureg.parse_expression(self.flowrateunits)).to(ureg.gpm).magnitude
            
            if not hasattr(self, 'coeffs_list'):
                self.result_label.text = 'Error: Please calculate coefficients first.'
                return
            
            # 1) Convert torque to differential pressure
            givendiffpsi = self.em * giventorque  # intercept is known to be 0
            
            # 2) Convert differential pressure and flow rate to RPM by interpolating the coefficients
            thusrpm = get_interpolated_value(givendiffpsi, givenflow, self.coeffs_list, self.flow_list)
            thusrpm = round(thusrpm,2)
            if givendiffpsi>self.stallpressure:
                self.result_label.text = f"Calculated RPM: Motor Stall"
            elif thusrpm>self.overspeed:
                self.result_label.text = f"Calculated RPM: {thusrpm}: Overspeed"
            else:
                self.result_label.text = f"Calculated RPM: {thusrpm}"
        except ValueError:
            self.result_label.text = 'Error: Invalid input values.'

    async def on_save_to_database(self, widget):
        motor_id = self.mudmotor_id_input.value.strip()  # Get the motor ID from the input field

        if not motor_id:
            self.result_label.text = 'Error: Motor ID cannot be empty.'
            return

        def format_float(value, decimal_places=20):
            if isinstance(value, (float, np.float32, np.float64)):
                return f"{value:.{decimal_places}f}"
            elif isinstance(value, np.ndarray):
                return [format_float(v, decimal_places) for v in value]
            elif isinstance(value, list):
                return [format_float(v, decimal_places) for v in value]
            else:
                return value

        motor_data = {
            "coeffs_list": [format_float(coeff) for coeff in self.coeffs_list],
            "em": format_float(self.em),
            "flow_list": format_float(self.flow_list),
            "stallpressure": format_float(self.stallpressure),
            "overspeed": format_float(self.overspeed),
        }
        
        # Load existing data from motor_db.json if it exists
        if os.path.exists(database_path):
            with open(database_path, 'r') as f:
                database = json.load(f)
        else:
            database = {}

        # Update the database with the new or modified motor data
        database[motor_id] = motor_data

        # Save the updated database back to motor_db.json
        with open(database_path, 'w') as f:
            json.dump(database, f, indent=4)
        
        print(f'Motor data for ID {motor_id} saved successfully.')

def main():
    return CurveExtractorApp('MotorDB', 'in.rocklab.motorDB')

if __name__ == '__main__':
    main().main_loop()
