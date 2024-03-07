# Effects-of-a-dark-matter-caustic-passing-through-the-Oort-Cloud-code
Code for paper Effects of a dark matter caustic passing through the Oort Cloud.

## Envoriment version

The code is written and tested under the following envoriment:

- Platform: Windows 11 x64
- Python: 3.11.7
- numpy: 1.24.2
- scipy: 1.10.1
- matplotlib: 3.7.0
- p-tqdm: 1.4.0

## Useage

This code solves equations of motion of objects in the presence of dark matter caustics.

Example:
1. Adjust the control parameters in file 'caistoc_wall.py' and save.
2. Run the code. The code will write data in to files '\textit{your name} -config.bin' and '\textit{your name} -data.bin'.

        python3 caustic_wall.py

3. Open 'plot.py' and check if the 'fname' variable has the value '{your name}'. Run 'plot.py'. It will process raw data and prepare data ready to plot in '\textit{your name} -plot.bin'.

        python3 plot.py

4. Run Jupyter-notebook to plot.

## License

This project is licensed under the [MIT License](LICENSE).

