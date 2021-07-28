# DrawSaintMask
 Draw X-ray aperture mask (xa) files for the SCXRD data integration engine Saint
 
 ## Currently understands:
  - [Dectris](https://www.dectris.com/detectors/x-ray-detectors/pilatus3/) Pilatus3 (.tif)
  - [Bruker](https://www.bruker.com/en/products-and-solutions/diffractometers-and-scattering-systems/single-crystal-x-ray-diffractometers/sc-xrd-components/detectors.html) Photon II (.raw)
 
 ## What to do:
 - move and reshape a circle and a rectangle to cover the beamstop shadow
 
 ## Additionally:
 - mask circular regions
   - black circle masks the area it covers
   - green circle unmasks the area it covers
   - follows hierarchal order (smaller overwrites larger)
 - mask negative values on Pilatus Detectors
 - mask horizontal lines on a Bruker Photon II Detector
 
 ## Good to know:
 - objects are allowed to cover the image only partly
 - to mask a corner
   - cover the corner with a black circle
   - move the green circle outside of the image area
   - the mask will have the proper dimensions
 - use the slider to adjust the contrast
 - statusbar shows the intensity at the current mouse position x, y
 
 ## How it looks:
![Image](../main/assets/DrawSaintMask.png)
