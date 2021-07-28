# DrawSaintMask
 Draw X-ray aperture mask (xa) files for the SCXRD data integration engine SAINT
 
 Uses [PyQt5](https://www.riverbankcomputing.com/static/Docs/PyQt5/) & [PyQtGraph](https://www.pyqtgraph.org/)!
 
 ## Currently understands:
  - [Dectris](https://www.dectris.com/detectors/x-ray-detectors/pilatus3/) Pilatus3 (.tif)
  - [Bruker](https://www.bruker.com/en/products-and-solutions/diffractometers-and-scattering-systems/single-crystal-x-ray-diffractometers/sc-xrd-components/detectors.html) Photon II (.raw)
 
 ## What to do:
 - move and reshape a circle and a rectangle to cover the beamstop shadow
  
 ## Has buttons to:
 - mask negative values (Pilatus Detectors)
 - mask horizontal lines (Bruker Photon II Detector)
 - mask rings, corners etc.
 
 ## Add circular region masks
   - black circle masks the area it covers
   - green circle unmasks the area it covers
   - artistic arrangements are welcome
   - follows hierarchal order (small > large)
 
 ## Good to know:
 - the initial rectangle & circle will always be on top
 - objects are allowed to be placed anywhere
 - to mask a corner
   - cover the corner with a black circle (as illustrated below)
   - move the green circle outside of the image area (not necessary)
   - the mask will have the proper dimensions
 - use the slider to adjust the contrast
 - statusbar shows the intensity at the current mouse position x, y
 
 ## Can learn new formats:
  - currently needs:
    - SAINT detector name tag (e.g. 'CMOS-PHOTONII')
    - detector dimension (pixels)
    - header offset in bytes
    - datatype
    - rotation?
  - limited flexibility but nothing is impossible
  - check _add new detector formats_ in _DrawSaintMask.py_
 
 ## How it looks:
![Image](../main/assets/DrawSaintMask.png)

 ## The xa mask:
![Image](../main/assets/DrawSaintMask_xa.png)
