import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.morphology import binary_erosion, binary_dilation
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.optimize import fsolve


class LassoSelectorHandler:
    def __init__(self, ax, mask):
        self.ax = ax
        self.mask = mask
        self.selected_coords = None
        self.lasso = LassoSelector(ax, onselect=self.on_select)
        self.canvas = ax.figure.canvas

    def on_select(self, verts):
        path = Path(verts)
        y_indices, x_indices = np.where(self.mask)
        points = np.column_stack([x_indices, y_indices])
        self.selected_coords = points[path.contains_points(points)]
        self.lasso.disconnect_events()
        self.canvas.draw_idle()

def interactive_crop(img, crop_coords=None):
    if crop_coords is None:
        plt.imshow(img)
        plt.title("Pick corner points to crop the image (close to the axes)")
        crop_coords = plt.ginput(2)
        plt.close()

        if len(crop_coords) != 2:
            raise ValueError("Error: Exactly two points are required to define the crop area.")
        
        x1, y1 = map(int, crop_coords[0])
        x2, y2 = map(int, crop_coords[1])

        # Ensure coordinates are in correct order
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Ensure coordinates are within bounds
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])
    
    else:
        x1, y1 = crop_coords[0]
        x2, y2 = crop_coords[1]

    cropped_img = img[y1:y2, x1:x2]
    print(f"Crop coordinates: {crop_coords}")
    
    return cropped_img, (x1, y1), (x2, y2)

def morphological_curve_detection(mask, erosion_size=3, dilation_size=3):
    print(f"Initial mask has {np.sum(mask > 0)} non-zero pixels")

    mask = binary_erosion(mask, np.ones((erosion_size, erosion_size)))
    mask = binary_dilation(mask, np.ones((dilation_size, dilation_size)))
    
    print(f"Post-processed mask has {np.sum(mask > 0)} non-zero pixels")

    return mask

def polynomial(x, *coeffs):
    return sum(c * x**i for i, c in enumerate(coeffs))

def fit_curve(x_data, y_data, degree=5):
    coeffs, _ = curve_fit(lambda x, *p: polynomial(x, *p), x_data, y_data, p0=[1] * (degree + 1))
    return coeffs

def extract_curve(image_path, crop_coords, x_min, x_max, y_min, y_max, color_value=None, custom_x_values=[], lasso=None, ):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Step 1: Crop the image to focus on the area inside the axes
    cropped_img, crop_start, crop_end = interactive_crop(img, crop_coords)

    if color_value is not None:
        # Use the provided color value (programmatically assigned)
        color = np.array(color_value, dtype=np.uint8)
        print(f"Programmatically assigned color: {color}")
    else:
        # Manual color picking
        plt.imshow(cropped_img)
        click_event = plt.ginput(1)
        x, y = int(click_event[0][0]), int(click_event[0][1])
        color = cropped_img[y, x]
        print(f"Picked point color: {color}")

    # Increase color tolerance
    lower_bound = color - 30
    upper_bound = color + 30
    lower_bound = np.clip(lower_bound, 0, 255)
    upper_bound = np.clip(upper_bound, 0, 255)

    img_hsv = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HSV)
    color_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
    lower_bound_hsv = color_hsv - np.array([10, 50, 50])
    upper_bound_hsv = color_hsv + np.array([10, 50, 50])
    lower_bound_hsv = np.clip(lower_bound_hsv, 0, 255)
    upper_bound_hsv = np.clip(upper_bound_hsv, 0, 255)

    # Mask the image based on the assigned color range
    mask_rgb = cv2.inRange(cropped_img, lower_bound, upper_bound)
    mask_hsv = cv2.inRange(img_hsv, lower_bound_hsv, upper_bound_hsv)
    
    # Combine RGB and HSV masks
    mask = cv2.bitwise_or(mask_rgb, mask_hsv)
    print(f"Initial mask created with {np.sum(mask > 0)} non-zero pixels")
    
    # Apply morphological operations to clean up the mask
    mask = morphological_curve_detection(mask, 1, 1)
    
    # Step 4: Use the lasso tool to select the region of interest
    if lasso is None:
        # Use interactive lasso selection
        plt.figure()
        plt.title("Draw a lasso around the curve to select")
        ax = plt.gca()
        ax.imshow(mask, cmap='gray')
        lasso_handler = LassoSelectorHandler(ax, mask)
        plt.show()
        selected_coords = lasso_handler.selected_coords
        print(selected_coords)
        #plt.scatter(selected_coords.T[0],selected_coords.T[1],)
        #plt.show()
    else:
        # Example lasso points (replace with actual lasso points)
        lasso_points = np.array(lasso)
        lasso_points[:, 0] -= crop_start[0]
        lasso_points[:, 1] -= crop_start[1]
        # Create an empty mask
        lasso_mask = np.zeros(mask.shape[:2], dtype=np.uint8)

        # Create a polygon path from lasso points
        poly_path = Path(lasso_points)
        
        # Generate a grid of coordinates
        y, x = np.mgrid[:lasso_mask.shape[0], :lasso_mask.shape[1]]
        coords = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        # Check which coordinates are inside the polygon
        lasso_mask[poly_path.contains_points(coords).reshape(lasso_mask.shape)] = 1
        #plt.imshow(mask)
        #plt.show()
        # Apply the lasso mask to the original mask using element-wise multiplication
        masked_image = mask * lasso_mask
        #plt.imshow(masked_image)
        #plt.show()
        # Get the coordinates of non-zero pixels in the masked image
        selected_coords = np.argwhere(masked_image > 0)
        selected_coords = selected_coords[:, [1, 0]]
        #print(selected_coords)
        #plt.scatter(selected_coords.T[0],selected_coords.T[1],)
        #plt.show()
        
    if selected_coords is not None:
        print(f"Selected {len(selected_coords)} coordinates with the lasso tool")

        df = pd.DataFrame(selected_coords, columns=['x', 'y'])

        # Step 5: Convert the coordinates to unit scale
        xrb = float(crop_coords[1][0])
        xlb = float(crop_coords[0][0])
        yub = float(crop_coords[0][1])
        ylb = float(crop_coords[1][1])
        
        x_scale = float((x_max-x_min))/(xrb-xlb) #unit per pixel #psi per pixel
        y_scale = float((y_max-y_min))/(ylb-yub) #unit per pixel #rpm/torque per pixel
        
        df['x'] = ((df['x']*x_scale) + x_min)
        df['y'] = (y_max - (df['y']*y_scale))

        print(f"Converted coordinates to unit scale: x range [{x_min}, {x_max}], y range [{y_min}, {y_max}]")
        
        # Save to CSV
        csv_path = 'extracted_curve.csv'
        #df.to_csv(csv_path, index=False)
        print(f"Coordinates saved to {csv_path}")
        
        # Fit a curve to the data
        x_data = df['x'].values
        y_data = df['y'].values
        coeffs = fit_curve(x_data, y_data, degree=5)
        print(f"Fitted curve coefficients: {coeffs}")
        
        # Plot the fitted curve
        x_fit = np.linspace(x_min, x_max, 1000)  # More points for a smoother curve
        y_fit = polynomial(x_fit, *coeffs)
        
        plt.figure()
        plt.scatter(x_data, y_data, label='Extracted Data')
        plt.plot(x_fit, y_fit, color='red', label='Fitted Curve')
        plt.title("Curve Fitting")
        plt.legend()
        plt.show()
        
        if len(custom_x_values)>0:
            # Step 6: Convert input values from one axis to another using the fitted curve
            new_x_values = np.array(custom_x_values).astype(float)
            converted_y_values = polynomial(new_x_values, *coeffs)
            
            print("Given x values:", new_x_values)
            print("Converted y values:", converted_y_values)

            # Optional: Plot the conversion results with the fitted curve
            plt.figure()
            plt.scatter(new_x_values, converted_y_values, color='green', label='Converted Values')
            plt.plot(x_fit, y_fit, color='red', linestyle='-', label='Fitted Curve')
            plt.title("Axis Conversion")
            plt.xlabel('Input x values')
            plt.ylabel('Converted y values')
            plt.legend()
            plt.show()
        
        return coeffs

def interpolate_curves(coeffs_list, flow_rate, flow_rates):
    # Number of coefficients (assumes all curves have the same number of coefficients)
    num_coeffs = len(coeffs_list[0])
    
    # Create an interpolation function for each coefficient
    interpolated_coeffs = []
    for i in range(num_coeffs):
        # Extract the i-th coefficient from each set of coefficients
        coeffs = [coeffs[i] for coeffs in coeffs_list]
        
        # Create an interpolation function
        interpolator = interp1d(flow_rates, coeffs, kind='linear', fill_value='extrapolate')
        
        # Interpolate the i-th coefficient for the given flow rate
        interpolated_coeffs.append(interpolator(flow_rate))
    
    return interpolated_coeffs

def calculate_curve(coeffs, x_values):
    return polynomial(np.array(x_values).astype(float), *coeffs)

def interpolate_values(flow_rate, coeffs_list, flow_rates, x_values):
    interpolated_coeffs = interpolate_curves(coeffs_list, float(flow_rate), flow_rates)
    return calculate_curve(interpolated_coeffs, np.array(x_values, dtype=float))


def interpolate_multiple_curves(image_path, flow_rates,x_min, x_max, y_min, y_max, crop_coords=None, color_value=None):    
    # Extract coefficients for each curve
    coeffs_list = []
    for flow_rate in flow_rates:
        coeffs = extract_curve(image_path, crop_coords, x_min, x_max, y_min, y_max, color_value)
        coeffs_list.append(coeffs)
    return coeffs_list

# Function to get interpolated y value for any x and flow rate
def get_interpolated_value(x, flow_rate, coeffs_list, flow_rates):
    return calculate_curve(interpolate_curves(coeffs_list, flow_rate, flow_rates), x)

def rotate_image(image_path, direction='right'):
    # Read the image
    img = cv2.imread(image_path)
    if direction=='right':
        # Rotate the image 90 degrees clockwise
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    else:
        # Rotate the image 90 degrees clockwise
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Save the rotated image
    rotated_path = 'rotated_' + image_path
    cv2.imwrite(rotated_path, rotated_img)
    return rotated_path

def torque_to_pressure_conversion(image_path, crop_coords=None, x_max=1500, y_max=40000, point=None):
    def on_click(event):
        if event.button == 1:  # Left mouse button
            nonlocal picked_point
            picked_point = (event.xdata, event.ydata)
            plt.close()

    # Read and crop the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if crop_coords is None:
        cropped_img, (x1, y1), (x2, y2) = interactive_crop(img)
    else:
        x1, y1 = crop_coords[0]
        x2, y2 = crop_coords[1]
        cropped_img = img[y1:y2, x1:x2]

    # Calculate scaling factors
    scale_x = x_max / (x2 - x1)
    scale_y = y_max / (y2 - y1)

    # Calculate aspect ratio for proper display
    aspect_ratio = (x2 - x1) / (y2 - y1)
    
    if point is None:
        # Display the image in pressure vs torque coordinates
        fig, ax = plt.subplots(figsize=(10, 10 / aspect_ratio))
        ax.imshow(cropped_img, extent=[0, x_max, 0, y_max], aspect='auto')
        ax.set_xlabel('Pressure (psi)')
        ax.set_ylabel('Torque (Nm)')
        ax.set_title('Click a point on the torque curve')

        picked_point = None
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()

        if picked_point is None:
            raise ValueError("No point was picked")
    else:
        picked_point=point
    # Calculate slope (m) and intercept (c) for x = my + c
    # Note: we're fitting x as a function of y now
    m = picked_point[0] / picked_point[1]
    c = 0  # We assume the line passes through (0, 0)

    print(f"Linear fit: x = {m}y + {c}")
    
    # Plot the fitted line
    y = np.linspace(0, y_max, 100)
    x = m * y + c
    
    plt.figure(figsize=(10, 10 / aspect_ratio))
    plt.imshow(cropped_img, extent=[0, x_max, 0, y_max], aspect='auto', alpha=0.5)
    plt.plot(x, y, 'r-', label='Fitted Line')
    plt.scatter([picked_point[0]], [picked_point[1]], color='blue', s=50, label='Picked Point')
    plt.xlabel('Pressure (psi)')
    plt.ylabel('Torque (ft.lbs)')
    plt.title('Pressure vs Torque with Fitted Line')
    plt.legend()
    plt.show()

    return m, c, [(x1, y1), (x2, y2)]


def convert_torque_to_pressure(torque, m, c):
    return m * torque + c

def get_linear_torque_coeffs(path,maxtorq,maxdelp,crop_coords=None,point=None):
    m, c, crop_coords = torque_to_pressure_conversion(path, crop_coords, x_max=maxdelp, y_max=maxtorq, point=point)
    return m,c,crop_coords



"""# Example Usage Non Linear Torque Model
#crop_coords = [(109, 18), (821, 522)]
x_min_torque, x_max_torque = 0.0, 40000.0  # Now this is the torque range
y_min_torque, y_max_torque = 0.0, 1500.0  # Now this is the differential pressure range
colormeimpressed = [230, 10, 40]
# Extract torque coefficients
#rotate_image("Powr-curve-mud-motor.jpg",'right')
torque_coeffs = extract_curve("Powr-curve-mud-motor.jpg", crop_coords, x_min_torque, x_max_torque, y_min_torque, y_max_torque, colormeimpressed, [0,10000, 20000, 30000,40000])
# Calculate differential pressure for given torque values
torquelist = [10000.0, 20000.0, 30000.0]
dP_torque = calculate_curve(torque_coeffs, torquelist)
print(f"Differential pressures for torques {torquelist}: {dP_torque}")
"""
def get_interpolated_rpms(path,maxflow,minflow,maxrpm,minrpm,maxdelp,mindelp,crop_coords=None,color_value=None):
    flow_rates = [maxflow,minflow]
    x_min, x_max = mindelp, maxdelp  # Example x-axis data range
    y_min, y_max = minrpm, maxrpm  # Example y-axis data range

    coeffs_list = interpolate_multiple_curves(path,flow_rates,x_min, x_max, y_min, y_max, crop_coords, color_value)
    return coeffs_list

if __name__ == "__main__":
    # Example useage: Torque and Flowrate to RPM
    maxRPM = 150
    minRPM = 0
    maxgpm = 1200
    mingpm = 600
    maxtorque = 40000
    mintorque=0
    maxdelP = 1500
    mindelP = 0
    flist = [maxgpm,mingpm]
    crop_coords=[(109, 20), (819, 520)]
    color_value=[6, 60, 120]

    inputs = np.array([[250,11236],[340,14235],[426,20224],[634,33456],[400,32194]]) # Inputs are in [GPL,ft.lbs]
    lasso_points = np.array([(164, 133.125), (160, 131.125), (158, 127.125), (154, 125.125), (154, 123.125), 
                         (154, 119.125), (154, 117.125), (160, 113.125), (168, 111.125), (170, 109.125), 
                         (174, 109.125), (176, 107.125), (178, 105.125), (180, 103.125), (182, 103.125), 
                         (186, 103.125), (188, 103.125), (194, 103.125), (200, 103.125), (206, 103.125), 
                         (210, 103.125), (212, 103.125), (218, 103.125), (224, 103.125), (228, 103.125), 
                         (230, 103.125), (240, 103.125), (252, 103.125), (266, 107.125), (280, 109.125), 
                         (290, 113.125), (302, 115.125), (312, 115.125), (320, 119.125), (324, 119.125), 
                         (328, 119.125), (336, 121.125), (348, 125.125), (350, 125.125), (354, 125.125), 
                         (356, 125.125), (360, 125.125), (364, 127.125), (372, 127.125), (386, 131.125), 
                         (396, 133.125), (404, 133.125), (410, 137.125), (416, 137.125), (426, 137.125), 
                         (440, 137.125), (452, 137.125), (462, 137.125), (468, 137.125), (478, 137.125), 
                         (490, 139.125), (502, 143.125), (518, 145.125), (532, 147.125), (536, 149.125), 
                         (540, 149.125), (542, 149.125), (546, 149.125), (556, 151.125), (576, 155.125), 
                         (602, 161.125), (622, 163.125), (634, 167.125), (638, 167.125), (642, 167.125), 
                         (644, 167.125), (648, 169.125), (650, 169.125), (654, 169.125), (660, 175.125), 
                         (674, 181.125), (692, 187.125), (704, 193.125), (714, 197.125), (720, 199.125), 
                         (724, 203.125), (730, 207.125), (732, 209.125), (736, 211.125), (738, 211.125), 
                         (742, 215.125), (744, 217.125), (748, 219.125), (750, 221.125), (752, 223.125), 
                         (756, 227.125), (758, 227.125), (764, 229.125), (766, 229.125), (770, 231.125), 
                         (772, 233.125), (774, 235.125), (778, 239.125), (782, 245.125), (784, 249.125), 
                         (784, 251.125), (788, 259.125), (792, 265.125), (796, 267.125), (798, 271.125), 
                         (800, 273.125), (800, 275.125), (800, 279.125), (800, 281.125), (800, 285.125), 
                         (800, 287.125), (798, 291.125), (796, 293.125), (794, 293.125), (786, 293.125), 
                         (776, 291.125), (768, 291.125), (762, 289.125), (758, 285.125), (756, 285.125), 
                         (752, 285.125), (750, 285.125), (748, 285.125), (742, 283.125), (736, 281.125), 
                         (730, 279.125), (720, 277.125), (710, 273.125), (704, 271.125), (700, 269.125), 
                         (692, 265.125), (684, 261.125), (676, 259.125), (668, 255.125), (660, 253.125), 
                         (646, 247.125), (630, 241.125), (616, 235.125), (612, 235.125), (606, 235.125), 
                         (596, 231.125), (582, 225.125), (566, 219.125), (554, 213.125), (546, 211.125), 
                         (540, 207.125), (534, 205.125), (532, 205.125), (526, 201.125), (520, 199.125), 
                         (516, 199.125), (512, 195.125), (506, 193.125), (496, 191.125), (492, 189.125), 
                         (486, 189.125), (478, 187.125), (468, 187.125), (458, 183.125), (450, 183.125), 
                         (438, 183.125), (430, 181.125), (420, 181.125), (414, 181.125), (404, 181.125), 
                         (402, 181.125), (396, 181.125), (388, 177.125), (384, 177.125), (378, 177.125), 
                         (374, 175.125), (370, 175.125), (362, 175.125), (356, 171.125), (348, 171.125), 
                         (342, 169.125), (336, 169.125), (328, 165.125), (326, 165.125), (320, 165.125), 
                         (316, 163.125), (312, 163.125), (298, 161.125), (290, 159.125), (282, 157.125), 
                         (272, 157.125), (264, 153.125), (254, 151.125), (252, 151.125), (246, 151.125), 
                         (242, 151.125), (234, 147.125), (228, 147.125), (224, 147.125), (222, 147.125), 
                         (218, 147.125), (214, 147.125), (210, 147.125), (204, 147.125), (198, 147.125), 
                         (192, 149.125), (188, 149.125), (186, 149.125), (182, 147.125), (178, 145.125), 
                         (176, 145.125), (172, 143.125), (170, 139.125), (166, 137.125), (164, 133.125), 
                         (160, 131.125), (160, 129.125), (160, 125.125), (160, 123.125), (158, 121.125)])
    coeffs_lasso= extract_curve('Powr-curve-mud-motor.jpg',crop_coords,mindelP,maxdelP,minRPM,maxRPM,color_value,[],lasso_points)

    m, c, crop_coords = get_linear_torque_coeffs('Powr-curve-mud-motor.jpg',maxtorque,maxdelP,crop_coords)
    coeffs_list = get_interpolated_rpms('Powr-curve-mud-motor.jpg',maxgpm,mingpm,maxRPM,minRPM,maxdelP,mindelP,crop_coords,color_value)

    pds = convert_torque_to_pressure(inputs.T[1],m,c)
    gpms = inputs.T[0]
    RPMS = get_interpolated_value(pds, gpms, coeffs_list, flist)
    for i, rpm in enumerate(RPMS):
        print(f"Input GPM: {gpms[i]}, Torque: {inputs[i, 1]} ft.lbs => Calculated RPM: {rpm:.2f}")
    

    #def extract_curve(image_path, crop_coords, x_min, x_max, y_min, y_max, color_value=None, custom_x_values=[], lasso=None):
