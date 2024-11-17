import streamlit as st
import streamlit.components.v1 as stc
import base64
import json
from PIL import Image,ImageOps
from io import BytesIO
from extract_curves_fitted import b64toRGB, RGB2b64,get_linear_torque_coeffs,extract_curve
import pint

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


if 'rdata' not in st.session_state:
    st.session_state.rdata = None

# Declare the custom component
jsui = stc.declare_component('my_component', path='./components/jsui')

# Use the component
received_data = jsui()

if received_data is not None:
    # Your data from calculate() will be available here
    st.session_state.rdata = received_data
    data = json.loads(st.session_state.rdata)
    #st.write(data)
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
    #st.write(lasso_points_upper)
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
    st.write(coeffsUpper,coeffsLower)
