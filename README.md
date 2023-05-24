1. Install a version of Python between 3.6 and 3.7 directly on your system or in a python virtual environment or a conda environment.
2. Extract the energypredictor.zip folder
3. Open the python environment console (terminal/command prompt) and navigate to the extracted folder.<br/>
Example: cd C:\home\NCG\energypredictor
4. Using a text editor of choice, edit the homeFolderPath parameter in the <extracted-folder\config/config.env> file with the full path of the extracted folder (same path as in step 3).<br/> 
Example:  homeFolderPath=C:\home\NCG\energypredictor<br/> 
Save the file before closing.<br/>
5. Place the energy files NCG_commercial_conversion_final_data.xml  and NCG_external_balancing_gas.xml in the <extracted-folder\data\energy> folder. See the chapter on balancing energy data for information on how to get the energy data files.
6. If manually downloaded weather data will be used, place data_MO_RF_MN004.csv and other files in <extracted-folder\data\weather\manual>. See the chapter on weather data for information on how to get the weather data.<br/>
It is possible to use a different energy and weather data folder location and/or different data file names. The corresponding parameters would have to be changed in config.env.
7. Install the application by running the following on the python environment console (terminal/command prompt)<br/>
pip install dist\<name of the wheel file>.whl<br/>
Example:<br/> 
pip install C:\home\ncg\energypredictor\dist\energypredictor-0.1.0-py3-none-any.whl
8. Start the application by running the following on the python environment console (terminal/command prompt)
energy-predictor --env=<full path of config.env file><br/>
Example:<br/>
energy-predictor --env=C:\home\ncg\energypredictor\config\config.env
9. The application will automatically start in your default browser.

