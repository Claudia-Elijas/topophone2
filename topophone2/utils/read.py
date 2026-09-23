"""
.. container:: module-intro

    The ``topophone2.utils.read`` module provides functions to read the data and parese the configuration file.

    The main function is :func:`parse_config`, which reads the configuration file and returns a dictionary of variables.

    The following functions are used by :func:`parse_config` to parse specific entries in the configuration file:

        :func:`parse_array_from_config` - Parses a comma-separated string from the config file into a numpy array.

        :func:`read_csv_to_array` - Reads a csv file specified in the config file into a numpy array.

        :func:`sigs_input` - organises the signals and noise model inputs from the config file into a list of dictionaries to input to the inversion function.

    The following functions are used to read and write the input data files:

        :func:`read_txt` - Reads the frequency, x, and Gamma_obs arrays from a text file.

        :func:`convert_to_txt` - Converts the frequency, x, and Gamma_obs arrays to the suitable input text file for Topophone2.

        :func:`row` - Converts a 1D array of values to a comma-separated string with 17 significant digits.

"""

#________ IMPORT ________#
import configparser
import pandas as pd
import numpy as np
import os


# =========================================== # 
# configuration file parser 
# =========================================== # 

# ________FUNC to parse config.ini inputs ________
def parse_config(path_to_config_file):
    """
    Convert the configuration file to variables.

    Input
    -----

    path_to_config_file : ``str``
        Path to the configuration file

    Output
    ------
    inputs : ``dict``
        Dictionary containing all the variables read from the configuration file.

    """

    # read the configuration file ===============================
    config = configparser.ConfigParser()
    config.read(path_to_config_file)

    # read input data from path specified in config file =======
    # signal data
    data_input = config.get("data", "signal_data_path") 
    
    if not os.path.exists(data_input):
        raise FileNotFoundError(f">> Signal data file {data_input} does not exist.")
    freq_Hz, sensor_positions, Gamma_obs, N_obs = read_txt(data_input)

    # noise data
    noise_data_input = config.get("data", "noise_data_path", fallback=None)
    # if no input file is specified, set noise data to None
    if not noise_data_input:
        print("\nNo noise data file specified in config file. Setting Gamma_obs_noise to None.\n")
        Gamma_obs_noise = None
        N_obs_noise = None
    else:
        print(f"\nReading noise data from: {noise_data_input}\n")
        if not os.path.exists(noise_data_input):
            raise FileNotFoundError(f">> Noise data file {noise_data_input} does not exist.")
        _, _, Gamma_obs_noise, N_obs_noise = read_txt(noise_data_input)

    # read variables from file ==================================   
    angular_freq = 2 * np.pi * freq_Hz
    signals = config["signal_model"]["signals"].replace(" ", "").split(",")
    weight_units = config["signal_model"].get("weight_units", fallback="dB")

    signal_noise = config["noise_model"]["signal_noise"].replace(" ", "").split(",")[0]
    dim = config["noise_model"].getint("dim", fallback=None)
    s = config["noise_model"].getfloat("s", fallback=None)
    a = parse_array_from_config(config, "noise_model", "a")
    b = parse_array_from_config(config, "noise_model", "b")
    max_order = config["noise_model"].getint("max_order", fallback=None)
    sensor_selfnoise = config["noise_model"].getboolean("sensor_selfnoise", fallback=False)

    prior_bounds = read_csv_to_array(config, "posterior_pdf", "prior_bounds")

    N_flow = config["inversion"].getint("N_flow", fallback=None)
    mean_field = config["inversion"].getboolean("mean_field", fallback=False)
    ini_dist = config["inversion"].get("ini_dist", fallback="Uniform")
    learning_rate = config["inversion"].getfloat("learning_rate", fallback=None)
    N_iterations = config["inversion"].getint("N_iterations", fallback=None)
    N_montecarlo = config["inversion"].getint("N_montecarlo", fallback=None)
    N_out = config["inversion"].getint("N_out", fallback=None)
    verbose = config["inversion"].getboolean("verbose", fallback=False)
    N_samples = config["inversion"].getint("N_samples", fallback=None)

    inversion_outpath = config.get("save_out", "inversion_outpath", fallback=None) 
    model_outpath = config.get("save_out", "model_outpath")
    loss_outpath = config.get("save_out", "loss_outpath", fallback=None)

    # compose other variables ==========================================

    save_intermediate_result = False
    if inversion_outpath is not None:
        save_intermediate_result = True 

    custom_cf = {}

    sigs = sigs_input(signals, dim , signal_noise , a , b , s , max_order , sensor_selfnoise)


    # make dict to hold variables ==========================
    args = dict(
                Gamma_obs = Gamma_obs,
                Gamma_obs_noise = Gamma_obs_noise,
                N_obs = N_obs,
                N_obs_noise = N_obs_noise,
                x = sensor_positions,
                freq = angular_freq,
                sigs = sigs,
                weight_units = weight_units,
                custom_cf = custom_cf,
                prior_bounds = prior_bounds,
                N_flow = N_flow,
                mean_field = mean_field,
                ini_dist = ini_dist,
                learning_rate = learning_rate,
                outpath = inversion_outpath,
                N_iterations = N_iterations,
                N_montecarlo = N_montecarlo,
                N_out = N_out,
                save_intermediate_result = save_intermediate_result,
                verbose = verbose,
                N_samples = N_samples,
                model_outpath = model_outpath,
                loss_outpath = loss_outpath
            )

    return args

# functions used in parse_config to read arrays from config file and csv files ====================

# =========================================== # 
# pareser helper functions 
# =========================================== # 

# ____ FUNC to parse a comma-separated string from the config file into a numpy array
def parse_array_from_config(config, section, option):
    entry = config.get(section, option, fallback=None)
    if entry is None:
        #print(f"The optional entry '{option}' not found in section '{section}' of the config file. It is set to None.")
        return None    
    else: 
        string_list = entry.replace(" ", "").split(",")
        return np.array([float(x) for x in string_list])

# ___ FUNC to read a csv file specified in the config file into a numpy array
def read_csv_to_array(config, section, option):
    entry = config.get(section, option, fallback=None)
    if entry is None:
        #print(f"The optional entry '{option}' not found in section '{section}' of the config file. It is set to None.")
        return None
    else: 
        # check if the file exists
        file_path = entry
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        
        df = pd.read_csv(file_path, header=None)

        # raise an error if the file is emplty
        if df.empty:
            raise ValueError(f"File {file_path} is empty.")
    
        return df.values

# ___ FUNC to parse the signals and noise model inputs from the config file
def sigs_input(signals, dim , signal_noise , a , b , s , max_order , sensor_selfnoise):
    sigs = []
    sigs.extend(signals)
    if sensor_selfnoise: sigs.append('selfnoise')

    noise_dict = {'sig':signal_noise}
    if (signal_noise == 'isonoise'):
        if dim:
            noise_dict['dim'] = dim
        else:
            raise ValueError("The 'dim' parameter must be provided in the config file for 'isonoise'.")
    if s: 
        noise_dict['s'] = s
    if (signal_noise in ['noise2D', 'noise3D']):
        if (a is not None) and (b is not None):
            if len(a) != len(b):
                raise ValueError("The lengths of 'a' and 'b' must be the same.")
            noise_dict['a'] = a
            noise_dict['b'] = b
        elif max_order: 
            noise_dict['max_order'] = max_order
        else:
            raise ValueError("Either 'a' and 'b' or 'max_order' must be provided in the config file.")

    sigs.append(noise_dict)

    return sigs


# =========================================== # 
# read / write txt input files functions 
# =========================================== # 

# ________FUNC read txt input ________
def read_txt(input_file, order='C'):
    """
    Read the frequency, x, and Gamma_obs arrays from a text file.

    Input
    -----
    input_file : ``str``
        Path to the input text file

    order : ``str`` (default ``'C'``)
        Order in which Gamma_obs is flattened if no ``ORDER`` line is found in the file.
        ``'C'`` means receiver q varies fastest, then receiver p, then frequency

    Output
    ------
    frequency : ``numpy.ndarray``
        Frequency array
        (N_freq,)

    x : ``numpy.ndarray``
        Receiver positions array
        (N_rec, dims)

    Gamma_obs : ``numpy.ndarray``
        Transfer function array
        (N_freq, N_rec, N_rec)

    N_obs : ``int``
        Number of observations
    """

    print(f"\nReading data from: {input_file} ............................\n")
    with open(input_file, "r") as f:
        lines = f.readlines()

    # Read order
    # check if ORDER line exists, if not, use the provided order
    if any(line.startswith("ORDER") for line in lines):

        order_line = next(line for line in lines if line.startswith("ORDER"))
        order = order_line.split()[1].strip()
        
        if order not in ['C', 'F']:
            raise ValueError(f"Invalid order '{order}' found in {input_file}. Must be 'C' or 'F'.")

        print(f"ORDER line found in input file. Using order '{order}':")
        if order == 'C':
            print(">>> Reading Gamma_obs in 'C' order (receiver q varies fastest, then receiver p, then frequency).\n")
        if order == 'F':
            print(">>> Reading Gamma_obs in 'F' order (frequency varies fastest, then receiver p, then receiver q).\n") 

    else:
        print(f"WARNING: ORDER line NOT found in input file. Using provided order '{order}'.")

        if order not in ['C', 'F']:
                    raise ValueError(f"Invalid order '{order}' found in {input_file}. Must be 'C' or 'F'.")
    
        if order == 'C':
            print(">>> Reading Gamma_obs in 'C' order (receiver q varies fastest, then receiver p, then frequency).\n")
        if order == 'F':
            print(">>> Reading Gamma_obs in 'F' order (frequency varies fastest, then receiver p, then receiver q).\n")

    # Read number of observations
    if not any(line.startswith("NOBS") for line in lines): # check if NOBS line exists, if not, raise an error
        raise ValueError(f"NOBS line not found in {input_file}. Please include the number of observations (NOBS) in the input file.")
    
    nobs_line = next(line for line in lines if line.startswith("NOBS"))
    N_obs = int(nobs_line.split()[1])

    # Read frequency
    freq_line = next(line for line in lines if line.startswith("FREQ"))
    frequency = np.array([float(val) for val in freq_line.split()[1].split(",")])

    # Read receiver positions
    x_lines = [line for line in lines if any(line.startswith(label) for label in ["XPSN", "YPSN", "ZPSN"])]
    x = np.array([[float(val) for val in line.split()[1].split(",")] for line in x_lines]).T

    # Read CO, QUAD data
    co_quad_start = next(i for i, line in enumerate(lines) if line.startswith("CO,QUAD")) + 1
    co_quad_data = np.loadtxt(lines[co_quad_start:], delimiter=",")
    
    # getting the dimensions of the data
    N_freq = frequency.shape[0]  
    N_rec = x.shape[0]  
    N_dim = x.shape[1]

    print(f"Data info:")
    print(f">>> Number of frequencies (N_freq): {N_freq}")
    print(f">>> Number of receivers (N_rec): {N_rec}")
    if N_rec < 2:
        raise ValueError("At least 2 receivers are required.")
    print(f">>> Number of coordinate dimensions (N_dim): {N_dim}")
    print(f">>> Number of observations (N_obs): {N_obs}")

    # check that the number of rows in co_quad_data matches N_freq * N_rec * N_rec   
    if co_quad_data.shape[0] != N_freq * N_rec * N_rec:
        raise ValueError("Mismatch between CO, QUAD data and expected shape based on frequency and receiver positions.")

    # reshape co_quad_data into a complex array and then into Gamma_obs
    Gamma_obs_flat = co_quad_data[:, 0] + 1j * co_quad_data[:, 1]

    # reshape Gamma_obs_flat into (N_freq, N_rec, N_rec) using the specified order
    Gamma_obs = Gamma_obs_flat.reshape((N_freq, N_rec, N_rec), order=order)

    # check that Gamma_obs is Hermitian
    if np.allclose(Gamma_obs, np.conj(np.swapaxes(Gamma_obs, -1, -2)), atol=1e-8 * np.abs(Gamma_obs).max()):
        print("\nCHECKED: Gamma_obs is Hermitian.")
    else:
        raise ValueError("WARNING:Gamma_obs is not Hermitian.")

    print(f"\nData successfully read from: {input_file}.\n")

    return frequency, x, Gamma_obs, N_obs


# ________FUNC convert arrays to txt input file________
def convert_to_txt(frequency, x, Gamma_obs, output_path, N_obs, out_order='C'):
    """
    Convert the frequency, x, and Gamma_obs arrays to the suitable input text file for Topophone2.

    Input
    -----
    frequency : ``numpy.ndarray``
        Frequency array [Hz]
        (N_freq,)

    x : ``numpy.ndarray``
        Receiver positions array
        (N_rec, dims)

    Gamma_obs : ``numpy.ndarray``
        Array-covariance matrix
        (N_freq, N_rec, N_rec)

    output_path : ``str``
        Path to the output text file (``_topophone2.txt`` is appended)

    N_obs : ``int``
        Number of observations

    out_order : ``str`` (default ``'C'``)
        Order in which Gamma_obs is flattened. ``'C'`` means receiver q varies fastest,
        then receiver p, then frequency

    Output
    ------
    ``None``
        Writes a text file with the following layout::

            ORDER C
            NOBS n
            FREQ f1,f2,...
            XPSN x1,x2,...
            (YPSN y1,y2,...)
            (ZPSN z1,z2,...)
            CO,QUAD
            re111, im111
            re112, im112
            ...
    """

    # out_order 
    if out_order not in ['C', 'F']:
        raise ValueError("out_order must be 'C' or 'F'.")
   

    # Get the dimensions of the input arrays
    N_freq = frequency.shape[0] # number of frequency points
    N_rec = x.shape[0] # number of receiver positions
    N_dim = x.shape[1] # number of coordinate dimensions
   
    # make sure the frequency, x, and Gamma_obs arrays have compatible shapes
    if frequency.ndim != 1:
        raise ValueError("Frequency array must be 1D.")

    # check that Gamma_obs has 3 dims 
    if Gamma_obs.ndim != 3:
        raise ValueError("Gamma_obs array must be 3D.")

    # check that Gamma_obs.shape is N_freq x N_rec x N_rec
    if Gamma_obs.shape != (N_freq, N_rec, N_rec):
        raise ValueError(f"Gamma_obs shape {Gamma_obs.shape} does not match expected shape {(N_freq, N_rec, N_rec)}.")

    if N_dim > 3:
        raise ValueError("x can have at most 3 dimensions (XPSN, YPSN, ZPSN).")


    # tests done, print data info 
    print(f"Data info:")
    print(f">>> Number of frequency points (N_freq): {N_freq}")
    print(f">>> Number of receiver positions (N_rec): {N_rec}")
    print(f">>> Number of coordinate dimensions (N_dim): {N_dim}")
    print(f">>> Number of observations (N_obs): {N_obs}")

    
    # build the output file
    output_file = output_path + "_topophone2.txt"
     
    # 'C' order gives (freq, p, q) with q fastest: Re111, Re112, ..., Re121, ..., Re211, ...
    # so it is read back with CO.reshape(N_freq, N_rec, N_rec)
    if out_order == 'C':
        print("\nFlattening Gamma_obs in 'C' order (receiver q varies fastest, then receiver p, then frequency).")
    elif out_order == 'F':
            print("\nFlattening Gamma_obs in 'F' order (frequency varies fastest, then receiver p, then receiver q).")

    Gamma_obs_flat = Gamma_obs.reshape(-1, order=out_order)
    co_quad = np.column_stack([Gamma_obs_flat.real, Gamma_obs_flat.imag])
    with open(output_file, "w") as f:

        # write the order of the flattened Gamma_obs
        f.write(f"ORDER {out_order}\n\n")

        # write the number of observations
        f.write(f"NOBS {N_obs}\n\n")

        # write the frequency and receiver positions
        f.write(f"FREQ {row(frequency)}\n\n")

        # write the receiver positions
        for label, coords in zip(["XPSN", "YPSN", "ZPSN"], x.T):
            f.write(f"{label} {row(coords)}\n")

        # write the CO, QUAD data 
        f.write("\n\nCO,QUAD\n\n")
        np.savetxt(f, co_quad, delimiter=",", fmt="%.17g")

 
    print(f"\nData successfully written to: {output_file}")

# ________FUNC rows of Gamma_obs________
def row(values):
    """
    Function to convert a 1D array of values to a comma-separated string with 17 significant digits.

    """
    return ",".join(f"{v:.17g}" for v in values) # 17 digits: exact float64 round trip
