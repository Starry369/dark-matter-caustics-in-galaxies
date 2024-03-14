# Dark matter caustics in galaxies
This code solves equations of motion of objects in the presence of dark matter caustics.

## Envoriment version

Current version of the code is tested under the following envoriment:

- Platform: Microsoft Windows 11
- Python: 3.11.8
- numpy: 1.26.4
- scipy: 1.12.0
- matplotlib: 3.8.3
- p-tqdm: 1.4.0

## Useage
This code solves equations of motion of objects in the presence of dark matter caustics.

The 'caustic_wall.py' calculates the motion of an object close to the Sun (a comet) when a caustic surface of $A_2$ catastrophe type passes by and analyzes if the object will escape or fall within a certain range of the Sun.

Extensions may be made depending on future works.

Example:
1. Adjust the control parameters in file 'caistoc_wall.py' and save.
2. Run the code. The code will write data in to files 'your name -config.bin' and 'your name -data.bin'.

        python3 caustic_wall.py

3. Open 'plot.py' and check if the 'fname' variable has the value 'your name'. Run 'plot.py'. It will process raw data and prepare data ready to plot in 'your name -plot.bin'.

        python3 plot.py

4. Run Jupyter-notebook file 'plot.ipynb' to plot.

Some importane parameters of the code:
* 'caustic_wall.py'
    * Line 15-29: Control parameters including the properties of the caustic.
    * Line 171-211: Set integration intervals.
* 'plot.py'
    * Line 82: Decide what distance is close enough to be counted as fall in close to the Sun.

## Cite this code

If you believe this code is useful, please cite our paper. https://arxiv.org/abs/2403.06314

## License

This project is licensed under the [MIT License](LICENSE).

